#!/bin/bash

# Copy using rclone
## We need to set ONE thing for this
## Secrete key: xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
## Access key: xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
## Url: xxxxx.xxxxx.com
## Bucket name: object
## Set rclone config
# rclone config create <remote_name> s3 provider=Other access_key_id=<access_key_id> secret_access_key=<secret_access_key> endpoint=<endpoint_url>
LOCAL_PATH="/home/user/InstellaVL/playground/data/mosaic_shards/"
REMOTE_PATH="<remote_name>:object/"
TRANSFER_SPEED=1024
rclone copy $LOCAL_PATH $REMOTE_PATH --progress --transfers $TRANSFER_SPEED
