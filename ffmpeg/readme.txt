
            ______ ______                                  
           / ____// ____/____ ___   ____   ___   ____ _
          / /_   / /_   / __ `__ \ / __ \ / _ \ / __ `/
         / __/  / __/  / / / / / // /_/ //  __// /_/ /
        /_/    /_/    /_/ /_/ /_// .___/ \___/ \__, /
                                /_/           /____/


                build: ffmpeg-7.0.2-armel-static.tar.xz
              version: 7.0.2

                  gcc: 8.3.0
                 yasm: N/A
                 nasm: N/A

               libaom: N/A
               libass: 0.17.3
               libgme: 0.6.3
               libsrt: 1.4.4
               libvpx: 1.11.0
              libvmaf: N/A
              libx264: 0.164.x 
              libx265: N/A
              libxvid: 1.3.7 
              libwebp: 0.6.1 
              libzimg: N/A
              libzvbi: N/A
             libdav1d: N/A
            libgnutls: 3.7.2
            libtheora: 1.2.0alpha1+git
            libfrei0r: 1.6.1-2
           libvidstab: 1.20
          libfreetype: 2.9.1-3+deb10u1
          libharfbuzz: 3.1.1
          libopenjpeg: N/A 

              libalsa: 1.2.4
              libsoxr: 0.1.3
              libopus: 1.3.1
             libspeex: 1.2
            libvorbis: 1.3.7
           libmp3lame: 3.100 
        librubberband: N/A
       libvo-amrwbenc: 0.1.3-1+b1
    libopencore-amrnb: 0.1.3-2.1+b2
    libopencore-amrwb: 0.1.3-2.1+b2


     Notes:  A limitation of statically linking glibc is the loss of DNS resolution. Installing
             nscd through your package manager will fix this.

             The vmaf filter needs external files to work- see model/000-README.TXT


      This static build is licensed under the GNU General Public License version 3.

      
      Patreon:  https://www.patreon.com/johnvansickle
      Paypal:   https://www.paypal.me/johnvansickle 
      Ethereum: 0x491f0b4bAd15FF178257D9Fa81ce87baa8b6E242 
      Bitcoin:  3ErDdF5JeG9RMx2DXwXEeunrsc5dVHjmeq 
      Dogecoin: DH4WZPTjwKh2TarQhkpQrKjHZ9kNTkiMNL

      email: john.vansickle@gmail.com
      irc:   relaxed @ irc://chat.freenode.net #ffmpeg
      url:   https://johnvansickle.com/ffmpeg/
