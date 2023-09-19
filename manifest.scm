(specifications->manifest '("python"

                            ;; base packages
                            "bash-minimal"
                            "glibc-locales"
                            "nss-certs"
                            
                            ;; Common command line tools lest the container is too empty.
                            "coreutils"
                            "grep"
                            "git"
                            "make"
                            "zlib"
                            "nano"
                            
                            ;; python stuff
                            "python-toolchain"
                            )
                          )
