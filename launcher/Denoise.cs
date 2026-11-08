// Denoise.exe: starts the local web app, or the first-time setup when the runtime is not installed yet.
// Build: launcher\build.cmd (uses the C# compiler that ships with Windows' .NET Framework).
using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Windows.Forms;

[assembly: AssemblyTitle("Denoise")]
[assembly: AssemblyProduct("Denoise")]
[assembly: AssemblyDescription("Color-preserving cleanup of artifacts, ringing, moire and fish-scale texture")]
[assembly: AssemblyVersion("1.0.0.0")]
[assembly: AssemblyFileVersion("1.0.0.0")]

static class DenoiseLauncher
{
    static string Q(string s) { return "\"" + s + "\""; }

    [STAThread]
    static int Main()
    {
        string app = Path.GetDirectoryName(Application.ExecutablePath);
        string server = Path.Combine(app, "server.py");
        string pyw = Path.Combine(app, @"runtime\venv\Scripts\pythonw.exe");
        string ready = Path.Combine(app, @"runtime\ready.txt");
        string setup = Path.Combine(app, @"launcher\setup.ps1");
        if (!File.Exists(server) || !File.Exists(setup))
        {
            MessageBox.Show("找不到 server.py 或 launcher\\setup.ps1。\n请把 Denoise.exe 留在项目文件夹里运行（不要单独拷出来）。",
                "Denoise", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return 1;
        }
        if (File.Exists(pyw) && File.Exists(ready))
        {
            var run = new ProcessStartInfo(pyw, Q(server));
            run.WorkingDirectory = app;
            run.UseShellExecute = false;
            run.CreateNoWindow = true;
            Process.Start(run);  // the server opens the browser itself (or reopens the page if it is already running)
            return 0;
        }
        var answer = MessageBox.Show(
            "第一次运行需要准备运行环境：\n\n" +
            "· 识别显卡：NVIDIA 用 CUDA，支持的 AMD 显卡用 ROCm，其余用 CPU\n" +
            "· 下载 Python、PyTorch 和默认模型（约 0.5–3 GB，看显卡）\n\n" +
            "全部装在这个文件夹的 runtime 和 models 里，不改系统；删掉文件夹就是卸载。\n" +
            "安装窗口会显示进度，装完自动打开 Denoise。现在开始吗？",
            "Denoise 首次运行", MessageBoxButtons.OKCancel, MessageBoxIcon.Information);
        if (answer != DialogResult.OK) return 0;
        var install = new ProcessStartInfo("powershell.exe", "-NoProfile -ExecutionPolicy Bypass -File " + Q(setup));
        install.WorkingDirectory = app;
        install.UseShellExecute = true;  // its own console window, so the progress is visible
        Process.Start(install);
        return 0;
    }
}
