import boto3
import logging
import socket
import requests
from botocore.exceptions import ClientError
from botocore.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_private_connection():
    """
    Verify if we're using private AWS network by checking VPC endpoint DNS resolution
    """
    try:
        # Bedrock endpoint for us-east-1
        bedrock_endpoint = "bedrock.us-east-1.amazonaws.com"
        endpoint_ip = socket.gethostbyname(bedrock_endpoint)
        
        # Check if the resolved IP is private
        # Private IP ranges: 10.0.0.0/8, 172.16.0.0/12, 192.0.0.0/16
        is_private = any([
            endpoint_ip.startswith("10."),
            endpoint_ip.startswith("172.") and 16 <= int(endpoint_ip.split(".")[1]) <= 31,
            endpoint_ip.startswith("192.0.")
        ])
        
        if is_private:
            logger.info(f"Verified private connection. Endpoint {bedrock_endpoint} resolved to private IP: {endpoint_ip}")
        else:
            logger.warning(f"Public IP detected: {endpoint_ip}. Traffic might not be using VPC endpoints!")
            
        return is_private
    
    except Exception as e:
        logger.error(f"Error verifying private connection: {str(e)}")
        return False

def get_bedrock_client(region_name="us-east-1"):
    """
    Create a Bedrock client with VPC endpoint configuration
    """
    try:
        # Verify private connection first
        is_private = verify_private_connection()
        if not is_private:
            logger.warning("Traffic might be flowing over public internet!")

        # Configure the client with VPC endpoint settings
        config = Config(
            region_name=region_name,
            retries={'max_attempts': 3, 'mode': 'standard'},
            connect_timeout=5,
            read_timeout=10,
            # Use VPC endpoint URL if configured
            endpoint_url=f"https://bedrock.{region_name}.amazonaws.com"
        )

        # Create Bedrock client
        bedrock_client = boto3.client(
            service_name='bedrock',
            config=config
        )
        
        return bedrock_client
    
    except Exception as e:
        logger.error(f"Error creating Bedrock client: {str(e)}")
        raise

def list_foundation_models():
    """
    List all available and enabled foundation models in Bedrock.
    
    Returns:
        list: List of enabled model information
    """
    try:
        # Get Bedrock client
        client = get_bedrock_client()
        
        # List foundation models with pagination handling
        enabled_models = []
        paginator = client.get_paginator('list_foundation_models')
        
        for page in paginator.paginate():
            for model in page['modelSummaries']:
                if model.get('modelLifecycle', {}).get('status') == 'ACTIVE':
                    enabled_models.append({
                        'modelId': model['modelId'],
                        'provider': model['providerName'],
                        'inputModalities': model.get('inputModalities', []),
                        'outputModalities': model.get('outputModalities', [])
                    })
        
        return enabled_models
    
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        logger.error(f"AWS Error: {error_code} - {error_message}")
        raise
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise

def main():
    """
    Main function to execute the model listing and handle results.
    """
    try:
        logger.info("Starting to fetch enabled Bedrock models...")
        
        models = list_foundation_models()
        
        logger.info(f"Found {len(models)} enabled models:")
        for model in models:
            logger.info(f"Model ID: {model['modelId']}")
            logger.info(f"Provider: {model['provider']}")
            logger.info(f"Input Modalities: {', '.join(model['inputModalities'])}")
            logger.info(f"Output Modalities: {', '.join(model['outputModalities'])}")
            logger.info("-" * 50)
            
        return models
        
    except Exception as e:
        logger.error(f"Failed to list models: {str(e)}")
        raise

if __name__ == "__main__":
    main()
