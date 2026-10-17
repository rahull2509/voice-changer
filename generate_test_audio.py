import subprocess
import os

ps_script = """
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.Rate = 0
$s.SetOutputToWaveFile('app/ui/test_speech.wav')
$s.Speak('Namaste! Yeh VaniFlow Indian female voice ka live test sample hai.')
$s.Dispose()
Write-Host "Audio generation finished"
"""

with open("scratch_gen.ps1", "w", encoding="utf-8") as f:
    f.write(ps_script)

res = subprocess.run(["powershell", "-ExecutionPolicy", "Bypass", "-File", "scratch_gen.ps1"], capture_output=True, text=True)
print("STDOUT:", res.stdout)
if os.path.exists("scratch_gen.ps1"):
    os.remove("scratch_gen.ps1")
