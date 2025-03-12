import os
from dotenv import load_dotenv
from simple_salesforce import Salesforce
import urllib.parse

def init_salesforce_test():
    """Initialize Salesforce connection using environment variables."""
    try:
        print("\n=== Debug Information ===")
        
        # Print current working directory
        print(f"Current working directory: {os.getcwd()}")
        
        # Check if .env file exists
        env_file = os.path.join(os.getcwd(), '.env')
        print(f"\n.env file exists: {os.path.exists(env_file)}")
        
        # Print raw environment variables before loading .env
        print("\nEnvironment variables before loading .env:")
        print(f"SF_USERNAME: {os.environ.get('SF_USERNAME', 'Not set')}")
        print(f"SF_PASSWORD: {'*' * len(os.environ.get('SF_PASSWORD', '')) if os.environ.get('SF_PASSWORD') else 'Not set'}")
        print(f"SF_SECURITY_TOKEN: {os.environ.get('SF_SECURITY_TOKEN', 'Not set')}")
        print(f"SF_DOMAIN: {os.environ.get('SF_DOMAIN', 'Not set')}")
        
        # Load environment variables
        print("\n1. Loading environment variables...")
        load_dotenv(override=True)
        
        # Print environment variables after loading .env
        print("\nEnvironment variables after loading .env:")
        print(f"SF_USERNAME: {os.environ.get('SF_USERNAME', 'Not set')}")
        print(f"SF_PASSWORD: {'*' * len(os.environ.get('SF_PASSWORD', '')) if os.environ.get('SF_PASSWORD') else 'Not set'}")
        print(f"SF_SECURITY_TOKEN: {os.environ.get('SF_SECURITY_TOKEN', 'Not set')}")
        print(f"SF_DOMAIN: {os.environ.get('SF_DOMAIN', 'Not set')}")
        
        # Get credentials from environment variables
        print("\n2. Reading credentials from environment...")
        username = os.getenv('SF_USERNAME')
        password = os.getenv('SF_PASSWORD')
        security_token = os.getenv('SF_SECURITY_TOKEN')
        domain = os.getenv('SF_DOMAIN', 'login')
        
        # URL encode the password to handle special characters
        encoded_password = urllib.parse.quote(password, safe='')
        print(f"\nPassword encoding:")
        print(f"   • Original length: {len(password)}")
        print(f"   • Encoded length: {len(encoded_password)}")
        print(f"   • First 3 chars (encoded): {encoded_password[:3]}...")
        
        # Clean up domain if it's a full URL
        if domain.startswith('http'):
            domain = urllib.parse.urlparse(domain).netloc
            if domain.startswith('login.'):
                domain = 'login'
        
        print("\nCredential Details:")
        print(f"   • Username found: {bool(username)}")
        if username:
            print(f"   • Username length: {len(username)}")
            print(f"   • Username value: {username}")
        
        print(f"\n   • Password found: {bool(password)}")
        if password:
            print(f"   • Password length: {len(password)}")
            print(f"   • First 3 chars (original): {password[:3]}...")
        
        print(f"\n   • Security Token found: {bool(security_token)}")
        if security_token:
            print(f"   • Token length: {len(security_token)}")
            print(f"   • Token value: {security_token}")
        
        print(f"\n   • Domain: {domain}")
        
        # Calculate and show the potential URLs that will be used
        print("\nPotential Connection URLs:")
        print(f"   • If using login domain: https://login.salesforce.com")
        print(f"   • If using test domain: https://test.salesforce.com")
        print(f"   • Current domain setting: https://{domain}.salesforce.com")
        
        if not all([username, password, security_token]):
            print("\n❌ Missing required Salesforce credentials in .env file")
            missing = []
            if not username:
                missing.append("SF_USERNAME")
            if not password:
                missing.append("SF_PASSWORD")
            if not security_token:
                missing.append("SF_SECURITY_TOKEN")
            print(f"   Missing variables: {', '.join(missing)}")
            return None
        
        print("\n3. Attempting Salesforce connection...")
        print("   • Creating Salesforce instance...")
        
        # Print exact connection parameters (excluding password)
        print("\nConnection Parameters:")
        print(f"   username: {username}")
        print(f"   password: {'*' * len(encoded_password)} (URL encoded)")
        print(f"   security_token: {security_token}")
        print(f"   domain: {domain}")
        print(f"   full URL that will be used: https://{domain}.salesforce.com")
        
        # Create instance with debug info
        print("\nAttempting to create Salesforce instance with these parameters...")
        
        # Try with URL encoded password
        try:
            sf = Salesforce(
                username=username,
                password=encoded_password,
                security_token=security_token,
                domain=domain
            )
            print("   • Connection successful with URL encoded password!")
        except Exception as e:
            print("\n❌ Failed with URL encoded password, trying original password...")
            # If that fails, try with original password
            sf = Salesforce(
                username=username,
                password=password,
                security_token=security_token,
                domain=domain
            )
            print("   • Connection successful with original password!")
        
        # If we get here, connection was successful
        print(f"   • Connected to instance: {sf.sf_instance}")
        print(f"   • API Version: {sf.sf_version}")
        print(f"   • Proxies used: {sf.proxies if hasattr(sf, 'proxies') else 'None'}")
        print("======================\n")
        return sf
    except Exception as e:
        print("\n❌ Error connecting to Salesforce:")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        if hasattr(e, 'content'):
            print(f"   Error content: {e.content}")
        print("======================\n")
        return None

def test_connection():
    print("Initializing Salesforce connection...")
    sf = init_salesforce_test()
    
    if sf:
        print("✅ Successfully connected to Salesforce!")
        
        # Test a simple SOQL query
        print("\nTesting SOQL query...")
        test_query = """
            SELECT Id, CaseNumber, Subject, CreatedDate
            FROM Case
            LIMIT 5
        """
        
        try:
            print("\n=== Query Debug Information ===")
            print(f"1. Executing query:\n{test_query}")
            results = sf.query_all(test_query)
            
            print("\n2. Query response:")
            print(f"   • Total size: {results.get('totalSize', 'N/A')}")
            print(f"   • Done?: {results.get('done', 'N/A')}")
            
            if results.get('records'):
                print("✅ Successfully executed query!")
                print("\nSample results:")
                for record in results['records'][:2]:  # Show first 2 records
                    print(f"\nRecord details:")
                    print(f"   • Case Number: {record.get('CaseNumber')}")
                    print(f"   • Subject: {record.get('Subject')}")
                    print(f"   • Created Date: {record.get('CreatedDate')}")
                    print("---")
            else:
                print("❌ Query returned no results")
                
            print("======================\n")
        except Exception as e:
            print("\n❌ Error executing query:")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Error message: {str(e)}")
            if hasattr(e, 'content'):
                print(f"   Error content: {e.content}")
            print("======================\n")
    else:
        print("❌ Failed to connect to Salesforce")

if __name__ == "__main__":
    test_connection() 