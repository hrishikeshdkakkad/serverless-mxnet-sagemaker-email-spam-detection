import json
import boto3
import os
import io
import email
import numpy as np
from botocore.exceptions import ClientError
from sms_spam_classifier_utilities import one_hot_encode
from sms_spam_classifier_utilities import vectorize_sequences

runtime= boto3.client('runtime.sagemaker')
ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
s3 = boto3.client("s3")

def lambda_handler(event, context):

    file_obj = event["Records"][0]
    filename = str(file_obj["s3"]['object']['key'])
    #print("filename: ", filename)
    fileObj = s3.get_object(Bucket = "spam-emails", Key=filename)
    #print("File:", fileObj)
    msg = email.message_from_bytes(fileObj['Body'].read())
    from_email = msg.get('From')
    subject = msg['Subject']
    body = msg.get_payload()[0].get_payload()
    #print("Body: ", body)
    
    label, score = predictspam(body)
    send_notification(from_email, body, subject, label, score)
    
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }
    

def predictspam(body):
    vocabulary_length = 9013

    #test_messages = ["FreeMsg: Txt: CALL to No: 86888 & claim your reward of 3 hours talk time to use from your phone now! ubscribe6GBP/ mnth inc 3hrs 16 stop?txtStop"]
    
    one_hot_test_messages = one_hot_encode(body, vocabulary_length)
    encoded_test_messages = vectorize_sequences(one_hot_test_messages, vocabulary_length)
    data = json.dumps(encoded_test_messages.tolist())

    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                       ContentType="application/json",
                                       Body=data)
    # print("Response: ", response)
    # result = response["Body"].read()
    # print("Result: ", result)
    # pred = int(result["predictions"][0]["score"])
    # print("Prediction: ", pred)
    # return pred
    
    res = json.loads(response["Body"].read())
    print("Prediction: ", res['predicted_label'])
    if res['predicted_label'][0][0] == 0:
        label = 'Ham'
    else:
        label = 'Spam'
    score = round(res['predicted_probability'][0][0], 4)
    return label, score*100

def send_notification(from_email, body, subject, label, score):

    SENDER = "hrishidkakkad@gmail.com"
    
    RECIPIENT = from_email
    
    # If necessary, replace us-west-2 with the AWS Region you're using for Amazon SES.
    AWS_REGION = "us-west-2"
    
    # The subject line for the email.
    SUBJECT = "Hello from lambda"
    
    # The email body for recipients with non-HTML email clients.
    BODY_TEXT = "We received your email with the subject "+ subject + ".\n" + "Here is a 240 character sample of the email body: "+ body[:240] + "\n" + "The email was categorized as "+ label  +" with a "+ str(score) + "% confidence."
                
                
    # The HTML body of the email.
    BODY_HTML = """<html>
    <head></head>
    <body>
      <h1>Amazon SES Test (SDK for Python)</h1>
      <p>This email was sent with
        <a href='https://aws.amazon.com/ses/'>Amazon SES</a> using the
        <a href='https://aws.amazon.com/sdk-for-python/'>
          AWS SDK for Python (Boto)</a>.</p>
    </body>
    </html>
                """            
    
    # The character encoding for the email.
    CHARSET = "UTF-8"
    
    # Create a new SES resource and specify a region.
    client = boto3.client('ses',region_name=AWS_REGION)
    
    try:
        #Provide the contents of the email.
        response = client.send_email(
            Destination={
                'ToAddresses': [
                    RECIPIENT,
                ],
            },
            Message={
                'Body': {
                    # 'Html': {
                    #     'Charset': CHARSET,
                    #     'Data': BODY_HTML,
                    # },
                    'Text': {
                        'Charset': CHARSET,
                        'Data': BODY_TEXT,
                    }
                },
                'Subject': {
                    'Charset': CHARSET,
                    'Data': SUBJECT,
                },
            },
            Source=SENDER
        )
    except ClientError as e:
        print(e.response['Error']['Message'])
    else:
        print("Email sent! Message ID:"),
        print(response['MessageId'])