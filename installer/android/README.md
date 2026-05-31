### Build for Android

##### Install dependencies

Several system tools and libraries are required for building and compiling project file binaries. For Android, there are also a few additional requirements:

        make system_dependencies



##### Prepare keystore to protect your .APK

To publish the app at Google Play Market, the .APK/.AAR file must be digitally signed.

First create a keystore file:

        mkdir ~/keystores/
        keytool -genkey -v -keystore ~/keystores/app.keystore -alias app -keyalg RSA -keysize 4096 -validity 60000


Ensure you have a backup copy of the keystore file and the keystore password!

Now you need to get "Encryption Key" from Google Play Console which you will use to prepare the `output.zip` file.

You only need to do this once. Also, the `output.zip` file must be uploaded back to Google. This way Google can verify the .APK/.AAR file you built before publishing it on the Play Market:

        java -jar pepk.jar --keystore=~/keystores/app.keystore --alias=app --encryptionkey=<Encryption Key> --include-cert --output=output.zip



##### Build AAR bundle file

        make



##### Build APK bundle file

        make apk
