# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 10:20:22 2021

@author: jgharris
"""
from model import *
import unittest
class TestDocClf(unittest.TestCase):
    def setUp(self):
       
        self.myModel=pickle.load(file=
            open('C:/Users/jgharris/DocClass/test/v2.pckmdl',"rb"))
        self.xtest='133d46f7ed38 96e70b2e2fc0 586242498a88 cde4f1b2a877 6bf9c0cb01b4 0562c756a2f2 fbb5efbcc5b3 d4e08985be1b 78c3a5c15b68 9ad186d42f69 6b343f522f78 bad6ff5dd7bc c337a85b8ef9 40f3a08093c6 65f888439937 2784a2673880 2272194f2c48 0969e9a2a900 9f11111004ec 1015893e384a 5e99d31d8fa4 d18e8c6fa60a c73d303c1dcb dc83f2b00468 24c356d262ce 818a7ff3bf29 578830762b27 9dc34464aa01 2e33ce9d2c13 e9f1f22efed4 fe081ae57a8b 7d9e333a86da ba02159e05b1 892d541c89eb 0e17d653f006 cf2205dbb077 fa736fae0d12 6b223a390d86 6ce6cc5a3203 2396ef1fa71b efaed70caea1 b208ae1e8232 c18666450d27 7ec02e30a5b3 ba3c06a73274 fdf32f896cc3 f36e139d9400 bf39bb85076d 1641a72fa752 1015893e384a 586242498a88 3012dd989e4f 6bf9c0cb01b4 041a934b1778 b32153b8b30c 95ef80a0b841 93790ade6682 a65259ff0092 7d9e333a86da e432c4501410 b61f1af56200 fe081ae57a8b d1dc04ffccd4 63fa15c4caa9 b136f6349cf3 8a3fc46e34c1 6d25574664d2 9cdf4a63deb0 b59e343416f7 6b7dd05245fd 0562c756a2f2 d774c0d219f8 6d40ee0e021c f02c3660e718 0562c756a2f2 4e5019f629a9 036087ac04f9 eeb86a6a04e4 98d0d51b397c ef237222adfc 1450168c4ac2 22b937b93ed2 f6f466726339 395a5e8185f8 07b4174549d4 b9925442c9c9 bfeaa5b4f65a ef4ba44cdf5f 9a42ead47d1c 73801426ea65 0226fe922dd0 376aa3d8142d e1a726bff8dc ed5d3a65ee2d 26f768da5068 5cc9caec5d01 b32153b8b30c 6a01047db3ab be9f9e5522c9 6bf9c0cb01b4 6d1fb90988cf'
        self.correctresult='DELETION OF INTEREST'
        self.correctConf=0.8676
    def testPredict(self):
       
        output=self.myModel.predict([self.xtest])
        print(output)
        assert(output[0]==self.correctresult)
    def testConfidence(self):
        output=self.myModel.predict([self.xtest])
        conf=self.myModel.getConfidence(self.xtest,output[0])
        self.assertAlmostEqual(conf,self.correctConf,delta=.001)
if __name__=='__main__':
    unittest.main()
class TestDocClf2(unittest.TestCase):
    def setUp(self):
       
        self.myModel=pickle.load(file=
            open('C:/Users/jgharris/DocClass/test/v3test.pckmdl',"rb"))

        self.xtest= 'ba02159e05b1 2ea49cf89745 878460b4304e 17cff84f5862 eb28356f8482 f8b0c07e306c 19ce59d8cd2f e7f10ad56136 6365c4563bd1 21e314d3afcc 6c841693835d 3d04e8a08335 d38820625542 e04a09c87692 cc65f20e23da ff1c26ea0b6f 4e5019f629a9 6fd647aa15b8 1eee686272e3 e8faf9f3abee 5ee06767bc0f 8754554be158 d38820625542 1015893e384a 54709b24b45f 5e99d31d8fa4 d38820625542 fdb96e216207 48b038f638e7 cf4fc632eed2 a0b19f1dc88f 432bd6e0c08f 5c02c2aaa67b e04a09c87692 586242498a88 cc65f20e23da d38820625542 ea51fa83c91c ff1c26ea0b6f 1015893e384a 2bcce4e05d9d d0a6ba7c50bf 04503bc22789 036087ac04f9 1015893e384a 5be138559904 1d4249bb404a d85aeb8537e1 a31962fbd5f3 75440bb763a2 93790ade6682 61b7e0f00ffe 6ca2dd348663 4357c81e10c1 d38820625542 6b343f522f78 11d62d3598ce b5dcdac45c9f 1d02937f11d6 e943e5e5b779 d38820625542 eb28356f8482 a7d9f88a65fa b73e657498f2 5ee06767bc0f a31962fbd5f3 75440bb763a2 93790ade6682 61b7e0f00ffe 4357c81e10c1 2bcce4e05d9d 036087ac04f9 7d9e333a86da 73d8f0e46834 1fe62f2b2bff d1cdc70d1aa6 133d46f7ed38 7d9e333a86da 87b8193a0183 66eb3ab508a0 d38820625542 557ec6c63cf9 306fe91e0f51 69e10bcd0d9a c36b062de326 feff74bf5e2d d38820625542 9d83e581af4b a8f8905ff606 b20bb43f3cb9 d38820625542 eb28356f8482 1f9291ef5874 6b343f522f78 dc03273b214a fea862065b74 64a7a7b8dfb5 0e60c8ecc79d d38820625542 7d9e333a86da 73d8f0e46834 1fe62f2b2bff d1cdc70d1aa6 cc7a3f31ec54 6b304aabdcee 133d46f7ed38 3fc879c6be39 d38820625542 ce1f034abb5d 8933b4be20c8 564aaf0c408b 6fd647aa15b8 6365c4563bd1 46c88d9303da 13b9e31ae11b aa1ef5f5355f cc429363fb23 6ad02ec3ba8c 6b343f522f78 50e7ce91c1cc 036928e5f0cc 8f75273e5510 7ba4dfb385ae 6b304aabdcee 530fd2faa74a 479e6e2b5afc 8e064f091129 54709b24b45f e4dad7cb07b6 8e064f091129 cc429363fb23 6ad02ec3ba8c d464ee914de0 036928e5f0cc 8f75273e5510 659ab661ba6a d38820625542 6fd647aa15b8 e7b697a1cd25 64a7a7b8dfb5 6b304aabdcee 530fd2faa74a 0562c756a2f2 a0c020166d79 1b6d0614f2c7 ff1c26ea0b6f e43c4b6f2c61 8f75273e5510 7ba4dfb385ae 64a7a7b8dfb5 93790ade6682 8528d46054c4 4357c81e10c1 a31962fbd5f3 75440bb763a2 036087ac04f9 2bcce4e05d9d ff1c26ea0b6f e04a09c87692 cc65f20e23da 5609253e130b bad6ff5dd7bc 34d9f9c9793b 26f768da5068 83ea5c938ad0 acc6c3f60d05 699252d4cf38 15d210f337d8 1b6d0614f2c7 7bf4f79c3fd9 6dcc75eaf823 131357427ea2 fdb96e216207 54709b24b45f f2b0e028fe2c 586242498a88 306fe91e0f51 d38820625542 b73e657498f2 421e52f8278f f68997f9c871 918d14133622 48d657cd9861 cf4fc632eed2 5ee06767bc0f 5ee06767bc0f 816a114f1b9a 060ddd403721 a4ab6e2c0f96 a263b63bc282 92335da34e06 1ab34730c1e0 7d9e333a86da 54709b24b45f 6ce6cc5a3203 9d17dd435ac4 376aa3d8142d abe7d2dd7c9b 43af6db29054 042293b11241 9b437b286e72 43af6db29054 6b304aabdcee 1b8aa9e2c997 bc74bd520012 418c0a4c2dbd da23520b125e d6d43b0803e9 fa80b6ed74f1 07c59f8e2e85 5948001254b3 7efa343de8e8 0fa1a6e5fd20 6f3bb46776c7 022341db1d4d cc429363fb23 e948abae8e94 459ec150cb83 6a95ce91efbd 6d8a803179cc 43af6db29054 6a95ce91efbd 5401b2de7490 6b304aabdcee df781fd33f77 71144e880c8d 64186c14a6be 047397193d3e f1424da4e7d6 8f75273e5510 507abf029e97 8b55de88409f 6b304aabdcee 294ef1562672 dcf7094a69ef 8bc44d97373a ef0a257522f5 52d07e9a5a47 0c4ce226d9fe f95d0bea231b 26f768da5068 9fc097bdc653 6a95ce91efbd c635c1efb452 04ea63a23e2a f733c1875bec 6b304aabdcee ba02159e05b1 cc429363fb23 5a7a442d189e 6af770640118 a7695ff8a61f 1068682ce752 1068682ce752 3689f1b1fc46 aa1ef5f5355f aa9714e02e29 c9ec91e2992f a47aade6bdf6 43565b1afa44 139eaf5e0ec4 80b9bc790801 bad6ff5dd7bc dc83f2b00468 e504ee0aaf6d 7bf4f79c3fd9 0d767bd685e2 8a3f710090fc b8c9830384f6 feb26a332310 65df965d2089 d8afd84c6fa9 e43c4b6f2c61'
        self.xtest3='0123456789012345678901234567890123456789'
        self.xtest2='586242498a88 cde4f1b2a877 6bf9c0cb01b4 0562c756a2f2'
        
        self.correctresult='BINDER'
        self.correctConf=0.466
       
        print(self.myModel.message)
    def testPredict(self):
       
        output=self.myModel.predict([self.xtest])
        print(output)
        self.assertEqual(output[0],self.correctresult)
    def testPreprocess(self):
        xprocessed,xbegin=self.myModel.preprocess([self.xtest,self.xtest2])
        print("Checking preprocessing")
        print(xprocessed[0][-1])
        self.assertEqual(xprocessed[0][0],"b")
        self.assertEqual(xprocessed[0][-1],"1")
        self.assertEqual(xprocessed[1][-1],"0")
    def testConfidence(self):
        output=self.myModel.predict([self.xtest])
        conf=self.myModel.getConfidence(self.xtest,output[0])
        self.assertAlmostEqual(conf,self.correctConf,delta=.001)
if __name__=='__main__':
    unittest.main()

