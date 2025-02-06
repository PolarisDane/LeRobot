import cv2
import os

def record_video():
    save_dir = './data/videos/test'
    cap = cv2.VideoCapture(12)
    out = None
    recording = False
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('frame', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            if not recording:
                out = cv2.VideoWriter(os.path.join(save_dir, f'output{idx}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (640, 480))
                recording = True
                print(f"Recording the {idx}-th demo")
            else:
                out.release()
                recording = False
                print(f"{idx}-th demo done recording")
                os.system("ffmpeg -i {} -r 10 {}".format(os.path.join(save_dir, f'output{idx}.mp4'), os.path.join(save_dir, f'{idx}.mp4')))
                os.system("rm {}".format(os.path.join(save_dir, f'output{idx}.mp4')))
                idx += 1

        if recording:
            out.write(frame)

        if key == ord('q'):
            break

    cap.release()
    if recording:
        out.release()
    cv2.destroyAllWindows()

record_video()