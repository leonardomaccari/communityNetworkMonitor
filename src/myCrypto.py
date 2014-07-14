
from Crypto.Cipher import AES
from Crypto.Hash import SHA256

class myCrypto:
    """ a simple class for encryption and decryption of data in the DB"""

    def __init__(self, key):
        """ class initialize, if no key is passed self.disabled = True """
        self.disabled = True
        if key != "":
            self.enabled = True
            h = SHA256.new(key)
            keyHash = []
            k = h.hexdigest()
            # transform an hex string into a sequence of chars
            for i in range(len(k))[::2]:
                # each couple of hex chars make a single digit
                keyHash.append(chr(int(k[i]+k[i+1], 16)))
            self.aes = AES.new("".join(keyHash))
            self.blockSize = 16

    def pad (self, s):
        """ return a string padded to the next block Size lenght """
        return s + (self.blockSize - len(s) % self.blockSize) \
                * chr(self.blockSize - len(s) % self.blockSize)

    def unpad (self, s):
        """ return a string unpadded to the previous block Size lenght """
        return s[0:-ord(s[-1])]

    def encrypt(self, s):
        """ use AES to encrypt content, encode in base64 to store it as string """
        if self.enabled:
            return str(self.aes.encrypt(self.pad(s))).encode('base64')
        else:
            return s

    def decrypt(self, s):
        """ decrypt and unpad the text """
        if self.enabled:
            return self.aes.decrypt(self.unpad(str(s.decode('base64'))))
        else:
            return s
