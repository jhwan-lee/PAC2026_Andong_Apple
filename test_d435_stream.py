"""
PAC 2026 안동청과 미션 - Intel RealSense D435 하드웨어 연결 및 RGB-D 정렬 검증 스크립트
================================================================================
기능:
1. D435 카메라 연결 및 시리얼 번호 / 펌웨어 정보 자동 인식
2. RGB(1280x720) + Depth(1280x720) 30fps 스트리밍
3. RGB-Depth 하드웨어 시차 자동 보정 (rs.align)
4. 화면 중앙 십자선 지점의 실시간 거리(mm) 계측 및 컬러맵 오버레이
5. 'q' 키를 누르면 종료

실행 방법:
  source /home/jaehwan/Documents/PAC2026_Andong_Apple/.venv/bin/activate
  python test_d435_stream.py
"""

import sys
import cv2
import numpy as np
import pyrealsense2 as rs

def main():
    print("=" * 60)
    print("🍎 [PAC 2026] Intel RealSense D435 스트리밍 테스트 시작")
    print("=" * 60)

    # 1. RealSense 파이프라인 및 설정 생성
    pipeline = rs.pipeline()
    config = rs.config()

    # 장치 감지
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        print("\n❌ RealSense 카메라가 감지되지 않았습니다!")
        print("  - D435 카메라를 USB 3.0 포트에 연결했는지 확인해주세요.")
        print("  - 'lsusb' 명령어로 Intel RealSense 장치가 보이는지 확인해주세요.")
        sys.exit(1)

    dev = devices[0]
    dev_name = dev.get_info(rs.camera_info.name)
    serial_no = dev.get_info(rs.camera_info.serial_number)
    fw_ver = dev.get_info(rs.camera_info.firmware_version)
    usb_type = dev.get_info(rs.camera_info.usb_type_descriptor)

    print(f"✅ 감지된 카메라: {dev_name}")
    print(f"   * 시리얼 번호: {serial_no}")
    print(f"   * 펌웨어 버전: {fw_ver}")
    print(f"   * USB 연결 모드: {usb_type}")

    if "3." not in usb_type:
        print("\n⚠️ [경고] 카메라가 USB 2.0으로 인식되었습니다!")
        print("   고해상도 RGB-D 동시 스트리밍을 위해 반드시 USB 3.0(파란색/SS) 포트에 연결해주세요.")

    # 스트림 활성화: 1280x720 @ 30fps
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    # 2. 스트리밍 시작
    try:
        profile = pipeline.start(config)
    except Exception as e:
        print(f"\n❌ 스트리밍 시작 실패 (해상도/대역폭 조정 필요 가능성): {e}")
        # 848x480 fallback 시도
        print("   848x480 안전 해상도로 재시도합니다...")
        config = rs.config()
        config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
        profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"   * 뎁스 스케일 계수: {depth_scale:.6f} m/unit")

    # RGB 프레임 기준으로 Depth를 정렬하는 객체 생성 (핵심!)
    align_to = rs.stream.color
    align = rs.align(align_to)

    # 컬러맵 시각화 객체
    colorizer = rs.colorizer()

    print("\n▶️ 실시간 영상 창을 띄웁니다. ('q' 키를 누르면 종료됩니다)")
    print("-" * 60)

    try:
        while True:
            # 프레임 세트 대기
            frames = pipeline.wait_for_frames()

            # RGB 기준으로 정렬
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not aligned_depth_frame or not color_frame:
                continue

            # Numpy 배열로 변환
            color_image = np.asanyarray(color_frame.get_data())
            depth_color_frame = colorizer.colorize(aligned_depth_frame)
            depth_color_image = np.asanyarray(depth_color_frame.get_data())

            h, w = color_image.shape[:2]
            cx, cy = w // 2, h // 2

            # 화면 중앙의 실제 깊이 거리(m) 측정
            center_dist = aligned_depth_frame.get_distance(cx, cy)
            dist_text = f"Center Dist: {center_dist * 1000:.1f} mm" if center_dist > 0 else "Center Dist: Out of Range"

            # 중앙 십자선 및 거리 텍스트 그리기
            cv2.drawMarker(color_image, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 30, 2)
            cv2.drawMarker(depth_color_image, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 30, 2)

            cv2.putText(color_image, dist_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(color_image, f"{dev_name} ({usb_type})", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 가로로 나란히 결합 (Side-by-Side)
            # 가로 해상도가 너무 클 경우 50% 축소
            preview_color = cv2.resize(color_image, (w // 2, h // 2))
            preview_depth = cv2.resize(depth_color_image, (w // 2, h // 2))
            combined = np.hstack((preview_color, preview_depth))

            cv2.imshow("Intel RealSense D435 [Color (Left) | Aligned Depth (Right)]", combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("\n⏹️ 스트리밍이 정상적으로 종료되었습니다.")

if __name__ == "__main__":
    main()
