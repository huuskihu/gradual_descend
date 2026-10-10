# Fix PATH: make D:\Program Files\python take precedence over MSYS2 python.
# ASCII-only on purpose (avoids encoding issues when run by elevated powershell).
# Must be run as Administrator (launched with -Verb RunAs).

$ErrorActionPreference = 'Stop'
$log = 'C:\Users\ASUS-PC\Desktop\files\code\ml\outputs\_path_fix_result.txt'
$backup = 'C:\Users\ASUS-PC\Desktop\files\code\ml\outputs\_path_backup_machine.txt'
$lines = @()

try {
    $m = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $m | Set-Content -Path $backup -Encoding UTF8
    $lines += "BACKUP saved to $backup"

    $pyRoot = 'D:\Program Files\python'
    $pyScripts = 'D:\Program Files\python\Scripts'
    $msys = 'D:\msys64\mingw64\bin'

    if (-not (Test-Path "$pyRoot\python.exe")) {
        $lines += "ERROR: python.exe not found in $pyRoot - aborting, nothing changed."
        $lines | Set-Content -Path $log -Encoding UTF8
        exit 1
    }

    $parts = @($m -split ';' | Where-Object { $_.Trim() -ne '' })

    # drop old python entries so we can re-insert them at the right position
    $parts = @($parts | Where-Object {
        ($_.TrimEnd('\') -ne $pyRoot) -and ($_.TrimEnd('\') -ne $pyScripts)
    })

    # find MSYS2 entry; insert python right before it (keeps msys64 in PATH)
    $idx = -1
    for ($i = 0; $i -lt $parts.Count; $i++) {
        if ($parts[$i].TrimEnd('\') -eq $msys) { $idx = $i; break }
    }

    $new = New-Object System.Collections.ArrayList
    if ($idx -ge 0) {
        for ($i = 0; $i -lt $parts.Count; $i++) {
            if ($i -eq $idx) { [void]$new.Add($pyRoot); [void]$new.Add($pyScripts) }
            [void]$new.Add($parts[$i])
        }
        $lines += "Inserted python entries before MSYS2 (index $idx)."
    } else {
        [void]$new.Add($pyRoot); [void]$new.Add($pyScripts)
        foreach ($p in $parts) { [void]$new.Add($p) }
        $lines += "MSYS2 entry not found; python entries prepended."
    }

    $newPath = ($new -join ';')
    [Environment]::SetEnvironmentVariable('Path', $newPath, 'Machine')
    $lines += "Machine PATH updated successfully."

    # notify running apps (Explorer etc.) that the environment changed
    try {
        Add-Type -Namespace Win32 -Name NativeMethods -MemberDefinition @'
[DllImport("user32.dll", SetLastError=true, CharSet=CharSet.Auto)]
public static extern IntPtr SendMessageTimeout(IntPtr hWnd, uint Msg, UIntPtr wParam, string lParam, uint fuFlags, uint uTimeout, out UIntPtr lpdwResult);
'@
        $HWND_BROADCAST = [IntPtr]0xffff
        $res = [UIntPtr]::Zero
        [void][Win32.NativeMethods]::SendMessageTimeout($HWND_BROADCAST, 0x001A, [UIntPtr]::Zero, "Environment", 2, 5000, [ref]$res)
        $lines += "WM_SETTINGCHANGE broadcast sent."
    } catch {
        $lines += "Broadcast failed (harmless): $($_.Exception.Message)"
    }

    $lines += "STATUS=OK"
} catch {
    $lines += "ERROR: $($_.Exception.Message)"
    $lines += "STATUS=FAILED"
}

$lines | Set-Content -Path $log -Encoding UTF8
