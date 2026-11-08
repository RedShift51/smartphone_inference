import subprocess, sys, os, boto3
from botocore.config import Config
def kc(a):
    return subprocess.run(["security","find-generic-password","-a",a,"-s",os.environ.get("R2_KEYCHAIN_SERVICE","r2-credentials"),"-w"],capture_output=True,text=True,check=True).stdout.strip()
s3=boto3.client("s3",config=Config(signature_version="s3v4"),region_name="auto",endpoint_url=f"https://{kc('R2_ACCOUNT_ID')}.r2.cloudflarestorage.com",aws_access_key_id=kc("R2_ACCESS_KEY_ID"),aws_secret_access_key=kc("R2_SECRET_ACCESS_KEY"))
bucket=kc("R2_BUCKET"); prefix, out = sys.argv[1], sys.argv[2]
n=0
for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket,Prefix=prefix):
    for o in page.get("Contents",[]):
        rel=o["Key"][len(prefix):]
        if not rel or rel.endswith("/"): continue
        dst=os.path.join(out,rel); os.makedirs(os.path.dirname(dst),exist_ok=True)
        if not os.path.exists(dst): s3.download_file(bucket,o["Key"],dst)
        n+=1
print("objects:",n)
