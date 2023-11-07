import os

from matplotlib import pyplot as plt, animation


def make_dir(*path_parts):
    dir_path = os.path.join(*path_parts)
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


class RecorderGif:
    def __init__(self, root_dir, time, agent, fps=120):
        self.save_dir = make_dir(root_dir, "logs/gifs/" + agent + "/" + time) if root_dir else None
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.time = time
        self.fps = fps
        self.frames = []
        pass

    def init(self):
        self.frames = []

    def animate(self, i):
        self.patch.set_data(self.frames[i])

    def recorder(self, step):
        self.patch = plt.imshow(self.frames[0])
        plt.axis('off')
        print("Generating GIF...")
        anim = animation.FuncAnimation(plt.gcf(), self.animate, frames=len(self.frames), interval=5)
        gif_path = os.path.join(self.save_dir, f"{step}.gif")
        try:
            anim.save(gif_path, writer='pillow', fps=self.fps, dpi=300)
            print(f"GIF saved at {gif_path}")
        except Exception as e:
            print(f"Error saving GIF: {e}")
        finally:
            plt.close()
        pass
