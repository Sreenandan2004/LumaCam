import os

manifest_path = "mobile/android/app/src/main/AndroidManifest.xml"
with open(manifest_path, "r", encoding="utf-8") as f:
    text = f.read()

snippet = """    <uses-permission android:name="android.permission.CAMERA" />
    <uses-feature android:name="android.hardware.camera" android:required="true" />
    <uses-feature android:name="android.hardware.camera.autofocus" android:required="false" />
    <uses-feature android:name="android.hardware.camera.raw" android:required="false" />

"""

if "android.permission.CAMERA" not in text:
    text = text.replace("<application", snippet + "    <application", 1)
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(text)
    print("Permissions successfully added to AndroidManifest.xml")
else:
    print("Permissions already present in AndroidManifest.xml")
