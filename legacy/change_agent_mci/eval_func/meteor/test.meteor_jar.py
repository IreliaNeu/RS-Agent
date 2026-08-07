import subprocess
import os


def test_meteor_jar():
    # 找到 meteor-1.5.jar 的完整路径
    meteor_dir = os.path.dirname(os.path.abspath(__file__))  # 当前目录
    meteor_jar = os.path.join(meteor_dir, "meteor-1.5.jar")

    if not os.path.exists(meteor_jar):
        print(f"[错误] 没有找到 meteor-1.5.jar 文件在: {meteor_jar}")
        return

    print(f"[信息] 找到 meteor-1.5.jar: {meteor_jar}")

    # 准备执行命令
    cmd = ['java', '-jar', meteor_jar, '-', '-', '-stdio', '-l', 'en', '-norm']

    try:
        print(f"[信息] 正在尝试启动 meteor.jar...")
        proc = subprocess.Popen(
            cmd,
            cwd=meteor_dir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1
        )
        print("[成功] Java进程已启动。")

        # 测试给Meteor发一条假指令，看看反应
        proc.stdin.write("EVAL ||| a ||| a\n")
        proc.stdin.flush()

        out = proc.stdout.readline()
        print(f"[输出] {out}")

        # 关闭
        proc.stdin.close()
        proc.kill()
        proc.wait()

    except Exception as e:
        print(f"[异常] 启动失败: {e}")


if __name__ == "__main__":
    test_meteor_jar()
