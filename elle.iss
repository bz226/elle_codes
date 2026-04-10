[Setup]
AppName=Elle
AppVerName=Elle 2.5
AppId=AppVerName
AppPublisherURL=http://www.microstructure.info/elle
AppSupportURL=http://www.microstructure.info/elle
AppUpdatesURL=http://www.microstructure.info/elle
DefaultDirName=c:\elle
DefaultGroupName=Elle
AllowNoIcons=yes
LicenseFile=C:\elle\LICENCE.txt
OutputDir=C:\elle_dist
OutputBaseFilename=ElleWinSetup
Compression=lzma
SolidCompression=yes
UsePreviousAppDir=yes
AlwaysShowComponentsList=no
ChangesEnvironment=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "dutch"; MessagesFile: "compiler:Languages\Dutch.isl"
Name: "french"; MessagesFile: "compiler:Languages\French.isl"
Name: "german"; MessagesFile: "compiler:Languages\German.isl"

[Tasks]
Name: desktopicon; Description: {cm:CreateDesktopIcon}; GroupDescription: {cm:AdditionalIcons}; Flags: unchecked

[Types]
Name: "User"; Description: "User Installation"
Name: "Developer"; Description: "Full Developer Installation"; Flags: iscustom


[Components]
Name: "src"; Description: "source code for processes and utilities " ; Types: Developer;
Name: "experiments"; Description: " experiments from the book " ; Types: Developer User;
Name: "extras"; Description: " sample elle files, ppm files and colormaps " ; Types: Developer User
Name: "executables"; Description: " executables for Windows " ; Types: Developer User
[Files]
 ;Source: "C:\elle\elle\binwx\*"; DestDir: "{app}\elle\binwx"; Components: executables; Flags: ignoreversion recursesubdirs createallsubdirs
 Source: "C:\elle\elle\binwx\basil2elle.exe"; DestDir: "{app}\elle\binwx"; Components: executables; Flags: ignoreversion
 Source: "C:\elle\elle\binwx\checkangle.exe"; DestDir: "{app}\elle\binwx"; Components: executables; Flags: ignoreversion
 Source: "C:\elle\elle\binwx\elle2poly.exe"; DestDir: "{app}\elle\binwx"; Components: executables; Flags: ignoreversion
 Source: "C:\elle\elle\binwx\elle_angle_rx.exe"; DestDir: "{app}\elle\binwx"; Components: executables; Flags: ignoreversion
 Source: "C:\elle\elle\binwx\elle_disloc_rx.exe"; DestDir: "{app}\elle\binwx"; Components: executables; Flags: ignoreversion
 Source: "C:\elle\elle\binwx\elle_exchange.exe"; DestDir: "{app}\elle\binwx"; Components: executables; Flags: ignoreversion
 Source: "C:\elle\elle\binwx\elle_expand.exe"; DestDir: "{app}\elle\binwx"; Components: executables; Flags: ignoreversion
 Source: "C:\elle\elle\binwx\elle_gbdiff.exe"; DestDir: "{app}\elle\binwx"; Components: executables; Flags: ignoreversion
 Source: "C:\elle\elle\binwx\elle_gbe.exe"; DestDir: "{app}\elle\binwx"; Components: executables; Flags: ignoreversion
 Source: "C:\elle\elle\binwx\elle_gbm.exe"; DestDir: "{app}\elle\binwx"; Components: executables; Flags: ignoreversion
 Source: "C:\elle\elle\binwx\elle_gg.exe"; DestDir: "{app}\elle\binwx"; Components: executables; Flags: ignoreversion
 Source: "C:\elle\elle\binwx\elle_latte.exe"; DestDir: "{app}\elle\binwx"; Components: executables; Flags: ignoreversion
 Source: "C:\elle\elle\binwx\elle_manuel.exe"; DestDir: "{app}\elle\binwx"; Components: executables; Flags: ignoreversion
 Source: "C:\elle\elle\binwx\elle_melt.exe"; DestDir: "{app}\elle\binwx"; Components: executables; Flags: ignoreversion
 Source: "C:\elle\elle\binwx\elle_metamorphism.exe"; DestDir: "{app}\elle\binwx"; Components: executables; Flags: ignoreversion
 Source: "C:\elle\elle\binwx\elle_nucleate.exe"; DestDir: "{app}\elle\binwx"; Components: executables; Flags: ignoreversion
 Source: "C:\elle\elle\binwx\elle_pblast.exe"; DestDir: "{app}\elle\binwx"; Components: executables; Flags: ignoreversion
 Source: "C:\elle\elle\binwx\elle_phasefield.exe"; DestDir: "{app}\elle\binwx"; Components: executables; Flags: ignoreversion
 Source: "C:\elle\elle\binwx\elle_proximity.exe"; DestDir: "{app}\elle\binwx"; Components: executables; Flags: ignoreversion
 Source: "C:\elle\elle\binwx\elle_recovery.exe"; DestDir: "{app}\elle\binwx"; Components: executables; Flags: ignoreversion
 Source: "C:\elle\elle\binwx\elle_split.exe"; DestDir: "{app}\elle\binwx"; Components: executables; Flags: ignoreversion
 Source: "C:\elle\elle\binwx\elle_stats.exe"; DestDir: "{app}\elle\binwx"; Components: executables; Flags: ignoreversion
 Source: "C:\elle\elle\binwx\elle_sub_gg.exe"; DestDir: "{app}\elle\binwx"; Components: executables; Flags: ignoreversion
 Source: "C:\elle\elle\binwx\elle_tbh.exe"; DestDir: "{app}\elle\binwx"; Components: executables; Flags: ignoreversion
 Source: "C:\elle\elle\binwx\elle_viscosity.exe"; DestDir: "{app}\elle\binwx"; Components: executables; Flags: ignoreversion
 Source: "C:\elle\elle\binwx\experiment_launcher.exe"; DestDir: "{app}\elle\binwx"; Components: executables; Flags: ignoreversion
 Source: "C:\elle\elle\binwx\examples.txt"; DestDir: "{app}\elle\binwx"; Components: executables; Flags: ignoreversion
 Source: "C:\elle\elle\binwx\single.txt"; DestDir: "{app}\elle\binwx"; Components: executables; Flags: ignoreversion
 Source: "C:\elle\elle\binwx\goof2elle.exe"; DestDir: "{app}\elle\binwx"; Components: executables; Flags: ignoreversion
 Source: "C:\elle\elle\binwx\ebsd2elle.exe"; DestDir: "{app}\elle\binwx"; Components: executables; Flags: ignoreversion
 Source: "C:\elle\elle\binwx\elle2ebsd.exe"; DestDir: "{app}\elle\binwx"; Components: executables; Flags: ignoreversion
 Source: "C:\elle\elle\binwx\lattestart.exe"; DestDir: "{app}\elle\binwx"; Components: executables; Flags: ignoreversion
 Source: "C:\elle\elle\binwx\lattestart.xrc"; DestDir: "{app}\elle\binwx"; Components: executables; Flags: ignoreversion
 Source: "C:\elle\elle\binwx\plotaxes.exe"; DestDir: "{app}\elle\binwx"; Components: executables; Flags: ignoreversion
 Source: "C:\elle\elle\binwx\poly2goof.exe"; DestDir: "{app}\elle\binwx"; Components: executables; Flags: ignoreversion
 Source: "C:\elle\elle\binwx\ppm2elle.exe"; DestDir: "{app}\elle\binwx"; Components: executables; Flags: ignoreversion
 Source: "C:\elle\elle\binwx\reposition.exe"; DestDir: "{app}\elle\binwx"; Components: executables; Flags: ignoreversion
 Source: "C:\elle\elle\binwx\showelle.exe"; DestDir: "{app}\elle\binwx"; Components: executables; Flags: ignoreversion
 Source: "C:\elle\elle\binwx\showelleps.exe"; DestDir: "{app}\elle\binwx"; Components: executables; Flags: ignoreversion
 Source: "C:\elle\elle\binwx\tidy.exe"; DestDir: "{app}\elle\binwx"; Components: executables; Flags: ignoreversion
 Source: "C:\elle\elle\binwx\triangle.exe"; DestDir: "{app}\elle\binwx"; Components: executables; Flags: ignoreversion
 Source: "C:\elle\elle\binwx\shelle24.html"; DestDir: "{app}\elle\binwx"; Components: executables; Flags: ignoreversion
 Source: "C:\elle\elle\binwx\sybil.bat"; DestDir: "{app}\elle\binwx"; Components: executables; Flags: ignoreversion
 Source: "C:\elle\elle\binwx\sybilps.bat"; DestDir: "{app}\elle\binwx"; Components: executables; Flags: ignoreversion
 Source: "C:\basil\bin\basil.exe"; DestDir: "{app}\elle\binwx"; Components: executables; Flags: ignoreversion
 Source: "C:\elle\elle\basecode\*"; DestDir: "{app}\elle\basecode"; Components: src; Flags: ignoreversion recursesubdirs createallsubdirs
 Source: "C:\elle\elle\statscode\*"; DestDir: "{app}\elle\statscode"; Components: src; Flags: ignoreversion recursesubdirs createallsubdirs
 Source: "C:\elle\elle\wxplotcode\*"; DestDir: "{app}\elle\wxplotcode"; Components: src; Flags: ignoreversion recursesubdirs createallsubdirs
 Source: "C:\elle\elle\plotcode\*"; DestDir: "{app}\elle\plotcode"; Components: src; Flags: ignoreversion  recursesubdirs createallsubdirs
 Source: "C:\elle\elle\processes\*"; DestDir: "{app}\elle\processes"; Components: src; Flags: ignoreversion  recursesubdirs createallsubdirs
 Source: "C:\elle\elle\utilities\*"; DestDir: "{app}\elle\utilities"; Components: src; Flags: ignoreversion  recursesubdirs createallsubdirs
 ; NOT INCLUDED 
 ;batchax batchps batch_oof
 ;elle_diff.exe make_exchng.exe
 ;elle_jkb-melt.exe elle_jkbgbe.exe elle_jkbgg.exe
 Source: "C:\mingw\bin\mingwm10.dll"; DestDir: "{app}\elle\binwx"; Components: executables; Flags: ignoreversion
 Source: "C:\elle\elle\docs\*"; DestDir: "{app}\elle\docs"; Flags: ignoreversion recursesubdirs createallsubdirs
 Source: "C:\elle\experiments\*"; DestDir: "{app}\experiments"; Components: experiments; Flags: ignoreversion recursesubdirs createallsubdirs
 Source: "C:\elle\extras\*"; DestDir: "{app}\extras"; Components: extras; Flags: ignoreversion recursesubdirs createallsubdirs
 Source: "C:\elle\COPYING.txt"; DestDir: "{app}"; Flags: ignoreversion
 Source: "C:\elle\LICENCE.txt"; DestDir: "{app}"; Flags: ignoreversion
 Source: "C:\elle\elle\elle.ico"; DestDir: "{app}\elle"; Flags: ignoreversion
 Source: "C:\elle\elle\Make.win"; DestDir: "{app}\elle"; Components: src; Flags: ignoreversion
 Source: "C:\elle\elle\install.win"; DestDir: "{app}\elle"; Components: src; Flags: ignoreversion
  ; NOTE: Don't use "Flags: ignoreversion" on any shared system files
  
  [INI]
 Filename: {app}\Internet shortcut.url; Section: "InternetShortcut"; Key: "URL"; String: "http://www.microstructure.info/elle"
  
  [Icons]
 Name: "{userdesktop}\Elle Experiments"; Filename: "{app}\elle\binwx\experiment_launcher.exe"; Tasks: desktopicon; IconFilename: "{app}\elle\elle.ico"
 Name: "{group}\Elle Experiments"; Filename: "{app}\elle\binwx\experiment_launcher.exe"; WorkingDir: "{app}\experiments"; IconFilename: "{app}\elle\elle.ico"
 Name: "{group}\{cm:ProgramOnTheWeb,Elle}"; Filename: "{app}\Internet shortcut.url"
 Name: "{group}\{cm:UninstallProgram,Elle}"; Filename: "{uninstallexe}"
  
  [UninstallDelete]
 Type: files; Name: {app}\Internet shortcut.url

 [Registry]
  Root: HKCU; SubKey: "Environment"; ValueType: string; ValueName: "ELLEPATH"; ValueData: "{app}\elle"; Flags: uninsdeletevalue
 Root: HKCU; SubKey: "Environment"; ValueType: string; ValueName: "PATH"; ValueData: "{app}\elle\binwx; {reg:HKCU\Environment,PATH}"

