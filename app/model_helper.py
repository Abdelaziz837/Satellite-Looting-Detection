import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import os
import boto3


# Validation transform (Resizes the uploaded image)
inference_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
# Define the classes
CLASS_NAMES = ["NotLooted", "Looted"]


class LootingClassifier:
    def __init__(self , model_paths: list):
        
        """ class for initianting the classifier 
         
            downloading the weights form the S3 Bucket 
                                                        """
        
        self.device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._download_from_s3(model_paths)

        self.ensemble_models = self._load_models(model_paths)

    def _download_from_s3(self , model_paths:list):
            """

            check for missing weights on local storage and securely download them from S3.
            does nothing if the files are already there (offline local testing)
            """

            bucket_name = os.getenv("S3_BUCKET_NAME")
            aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
            aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
            aws_region = os.getenv("AWS_REGION" , "us-east-1")

            missing_paths = [p for p in model_paths if not os.path.exists(p)]

            if not missing_paths:
                print("all models are present localy")
                return 
            
            if not (bucket_name and aws_access_key and aws_secret_key):
                raise ValueError(
                      f"Missing local weights for: {missing_paths}, "
                      "but S3 environment variables (S3_BUCKET_NAME, AWS_ACCESS_KEY_ID, "
                      "AWS_SECRET_ACCESS_KEY) are not configured."
                )
            
            print(f"connecting to s3 to dowenload missing weights : {missing_paths}")

            s3_client = boto3.client(
                "s3" , 
                aws_access_key_id = aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name = aws_region
            )

            for path in missing_paths:

                dir_name = os.path.dirname(path)  #create a local target folder
                if dir_name:
                    os.makedirs(dir_name , exist_ok=True)

                filename = os.path.basename(path)

                s3_key = filename

                try:
                    print(f"Dowenloading {filename} from s3 bucket '{bucket_name}' to '{path}'") 

                    s3_client.download_file(bucket_name, s3_key , path)
                    print(f"succesfully downloaded {filename}")

                except Exception as retry_error:
                    raise RuntimeError(
                        f"Failed to download {filename} from S3. Original Error: {e}. "
                        f"Root-retry Error: {retry_error}"
                    )
                
    def _load_models(self , model_paths :list) -> list :
            """
            takes the model paths and return it as a list
            """        
            
            ensemble_models = []

            for path in model_paths:
                model = models.resnet50(pretrained = False)

                nums_ftrs = model.fc.in_features
                model.fc = nn.Sequential(
                    nn.Dropout(p=0.4),
                    nn.Linear(nums_ftrs,2)
                )

                state_dict = torch.load(path, map_location=self.device)
                model.load_state_dict(state_dict)

                model.to(self.device)
                model.eval()
                ensemble_models.append(model)

            return ensemble_models

    def predict(self , image_bytes: bytes) -> dict:

            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            tensor = inference_transforms(image).unsqueeze(0).to(self.device) # type: ignore
            

            batch_probs = []  # list filled with prob of each model on the first class (looted)

            with torch.no_grad():  # turn off gradient descent

                for model in self.ensemble_models:
                    output = model(tensor)
                    probabilties = torch.softmax(output , dim = 1)[0]

                    batch_probs.append(probabilties)

                mean_prob = torch.mean(torch.stack(batch_probs,dim=0), dim=0)

                not_looted_prob = float(mean_prob[0])
                looted_prob = float(mean_prob[1])

                threshold = 0.4007

                predicted_class = "Looted" if looted_prob >= threshold else "NotLooted"
            
                return {
                    "prediction": predicted_class,
                    "probabilities": {
                       "NotLooted": round(not_looted_prob, 4),
                       "Looted": round(looted_prob, 4)
                },
                    "decision_threshold_applied": threshold
        }     




                
