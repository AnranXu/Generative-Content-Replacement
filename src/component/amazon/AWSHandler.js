import $ from "jquery";
import AWS from 'aws-sdk';

class AWSHandler{
    constructor(language='en', testMode=true)
    {
        AWS.config.region = 'ap-northeast-1'; 
        AWS.config.credentials = new AWS.CognitoIdentityCredentials({
            IdentityPoolId: 'ap-northeast-1:82459228-da79-4229-aeef-0168565a5e2e',
        });
        AWS.config.apiVersions = {
        cognitoidentity: '2014-06-30',
        // other service API versions
        };
        this.s3 = this.s3Init();
        this.db = this.dbInit();
        this.bucketName = 'diffusion-image-protection';
        //var len = 0;
    }
    s3Init() {
        var s3 = new AWS.S3({
            params: {Bucket: this.bucketName}
        });
        //const key = 'whole_.png';
        //var URIKey= encodeURIComponent(key);
        return s3;
    }   
    dbInit () {
        var dynamodb = new AWS.DynamoDB({apiVersion: '2012-08-10'});
        return dynamodb;
    } 
    uploadImage(userId, curNum, images, answers) {
        //upload to s3
        var res = JSON.stringify(answers);
        var textBlob = new Blob([res], {
            type: 'text/plain'
        });
        var answer_name = userId + '_' + curNum + '.json';
        this.s3.upload({
            Bucket: this.bucketName,
            Key: answer_name,
            Body: textBlob,
            ContentType: 'text/plain',
            ACL: 'bucket-owner-full-control'
        }, function(err, data) {
            if(err) {
                console.log(err);
            }
            }).on('httpUploadProgress', function (progress) {
            var uploaded = parseInt((progress.loaded * 100) / progress.total);
            $("progress").attr('value', uploaded);
        });

        for (var maskName in images) {
            var image_name = userId + '_' + curNum + '_' + maskName + '.png';
            var image_data = images[maskName];
            var imageBlob = new Blob([image_data], {
                type: 'image/png'
            });
            this.s3.upload({
                Bucket: this.bucketName,
                Key: image_name,
                Body: imageBlob,
                ContentType: 'image/png',
                ACL: 'bucket-owner-full-control'
            }, function(err, data) {
                if(err) {
                    console.log(err);
                }
                }
            ).on('httpUploadProgress', function (progress) {
                var uploaded = parseInt((progress.loaded * 100) / progress.total);
                $("progress").attr('value', uploaded);
            }
        );
        }


    }
}

export default AWSHandler;