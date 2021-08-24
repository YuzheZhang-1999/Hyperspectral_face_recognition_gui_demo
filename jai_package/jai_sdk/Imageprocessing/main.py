import ctypes
import numpy as np
import sys
import cv2
import requests
import base64
import json
import argparse

import urllib.request
import urllib.error
import time

from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.iai.v20200303 import iai_client, models

import whitebalance


#######################  baidu face  #################################
'''
baidu face recognition access token
client id = HzUb2pXnHCHvWL3MZQ4vPfqU
client secret = FTshQNKAWO87l1h3BGcWXrGtbIZABpep
'''
def getAccessToken(client_id, client_secret):
    host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id='+ client_id + '&client_secret='+ client_secret
    response = requests.get(host)
    if response:
        tokenResult = response.json()
        access_token = tokenResult['access_token']
        print(access_token)
        return access_token
    return ""


'''
transpose img to base64 type
'''
def picToBase64(file_path):
    with open(file_path, 'rb') as f:
        img = base64.b64encode(f.read()).decode('utf-8')
        return img

'''
get baidu api to recognize two faces
'''
def baidu_face(access_token, face1, face2):
    request_url = "https://aip.baidubce.com/rest/2.0/face/v3/match"
    params = [{"image": face1, "image_type": "BASE64", "face_type": "LIVE", "quality_control": "NORMAL", "liveness_control": "NORMAL"}, {"image": face2, "image_type":"BASE64", "face_type": "LIVE", "quality_control":"NORMAL", "liveness_control":"NORMAL"}]
    #params = json.dumps(params)
    access_token = access_token
    request_url = request_url + "?access_token=" + access_token
    headers = {'content-type': 'application/json'}
    response = requests.post(request_url, json=params, headers=headers)
    if response:
        response = response.json()
        print (response)
        if response['error_msg'] == 'SUCCESS':
            face_score = response['result']['score']
            print('face_socre:'+str(face_score))
            return face_score

############################# face ++ #####################################
'''
URL post
	https://api-cn.faceplusplus.com/facepp/v3/compare
call format
	multipart/form-data
'''

'''
Face++ face recognition access token
client id = Yjy2fTPkcJZBIhZ-HVt-xtipgZzHlO-6
client secret = BqkQ7L2ply2Cl485hB3K_SkyuB-Lz4T_
'''

# -*- coding: utf-8 -*-
def face_plusplus(image1, image2):
    http_url = 'https://api-cn.faceplusplus.com/facepp/v3/compare'
    key = "Yjy2fTPkcJZBIhZ-HVt-xtipgZzHlO-6"   
    secret = "BqkQ7L2ply2Cl485hB3K_SkyuB-Lz4T_"
    
    boundary = '----------%s' % hex(int(time.time() * 1000))
    data = []
    data.append('--%s' % boundary)
    data.append('Content-Disposition: form-data; name="%s"\r\n' % 'api_key')
    data.append(key)
    data.append('--%s' % boundary)
    data.append('Content-Disposition: form-data; name="%s"\r\n' % 'api_secret')
    data.append(secret)
    data.append('--%s' % boundary)
    data.append('Content-Disposition: form-data; name="%s"\r\n' % 'image_base64_1')
    data.append(image1)
    data.append('--%s' % boundary)
    data.append('Content-Disposition: form-data; name="%s"\r\n' % 'image_base64_2')
    data.append(image2)
    data.append('--%s--\r\n' % boundary)

    for i, d in enumerate(data):
        if isinstance(d, str):
            data[i] = d.encode('utf-8')

    http_body = b'\r\n'.join(data)

    # build http request
    req = urllib.request.Request(url=http_url, data=http_body)

    # header
    req.add_header('Content-Type', 'multipart/form-data; boundary=%s' % boundary)

    try:
        # post data to server
        resp = urllib.request.urlopen(req, timeout=5)
        # get response
        qrcont = resp.read()
        # if you want to load as json, you should decode first,
        # for example: json.loads(qrcont.decode('utf-8'))
        face_result = json.loads(qrcont.decode('utf-8'))
        print(face_result)
        print('face_socre:' + str(face_result['confidence']) + '\n')

    except urllib.error.HTTPError as e:
        print(e.read().decode('utf-8'))
    
    
############################# Tencent face #####################################
def tencent_face(image1, imgae2):
    key = "AKID60jETRTRhgPtHOTnBLEudAD7mCOJkMEs"   
    secret = "XperxsRmdHepoZ3GH4kAKI1Vt8B1cx63"      

    try: 
        cred = credential.Credential(key, secret) 
        httpProfile = HttpProfile()
        httpProfile.endpoint = "iai.tencentcloudapi.com"

        clientProfile = ClientProfile()
        clientProfile.httpProfile = httpProfile
        client = iai_client.IaiClient(cred, "ap-shanghai", clientProfile) 

        req = models.CompareFaceRequest()
        params = {"ImageA": image1,"ImageB": imgae2}
        req.from_json_string(json.dumps(params))

        resp = client.CompareFace(req) 
        print(resp.to_json_string()) 
        face_result = json.loads(resp.to_json_string())
        print('face_socre:' + str(face_result['Score']) + '\n')
    except TencentCloudSDKException as err: 
        print(err) 

########################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'face recognition method')
    parser.add_argument('--method', default='baiduface', help='choice which method to use: baiduface, face++, tencentface' )
    args = parser.parse_args()
    face_recognition_method = args.method

    lib=ctypes.cdll.LoadLibrary('./libImageOut.so')
    img=np.zeros((2048,2560,1),np.uint8)
    frame_data=np.asarray(img,dtype=np.uint8)
    frame_data=frame_data.ctypes.data_as(ctypes.c_char_p)

    lib.start()
     
    lib.acquireImages(frame_data)
    img = img.reshape(2048,2560,1)
    

    baidu_client_id = 'HzUb2pXnHCHvWL3MZQ4vPfqU'
    baidu_client_secret = 'FTshQNKAWO87l1h3BGcWXrGtbIZABpep'

    #cv2.imwrite('test.jpg',convert_img)
    while 1:
        convert_img = cv2.cvtColor(img,cv2.COLOR_BAYER_GR2RGB) 
        
        #convert_img = img
        #convert_img = cv2.pyrDown(convert_img)
        #convert_img = cv2.pyrDown(convert_img)
        '''
        try: 
            out_img=whitebalance.white_balance_5(out_img) 
        except:
            pass
        '''
        cv2.imshow('img',convert_img)
        if cv2.waitKey(1)&0xff == ord('s'): 
            cv2.imwrite('face_img/face1.jpg',convert_img)
          

        if cv2.waitKey(1)&0xff == ord('q'):             
            lib.stop()
            cv2.destroyAllWindows()

            if face_recognition_method == 'baiduface':
                print('Using Baidu Face \n')
                access_token = getAccessToken(baidu_client_id, baidu_client_secret)
                image1_base64 = picToBase64('face_img/base0.png')
                image2_base64 = picToBase64('face_img/face1.jpg')
                baidu_face(access_token, image1_base64, image2_base64)
            elif face_recognition_method == 'face++':
                print('Using Face++ \n')
                image1_base64 = picToBase64('face_img/base0.png')
                image2_base64 = picToBase64('face_img/base2.png')
                face_plusplus(image1_base64, image2_base64)
            elif face_recognition_method == 'tencentface':
                print('Using tencentface \n')
                image1_base64 = picToBase64('face_img/base0.png')
                image2_base64 = picToBase64('face_img/face1.jpg')
                tencent_face(image1_base64, image2_base64)
                
            else:
                print('No method has been choosed. \n')

            break



