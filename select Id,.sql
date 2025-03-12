select Id,
CaseNumber,
Account.Account_Name__c,
Eightfold_Group_Id__r.Group_Id__c, 
Subject,
Description,
Product_Area__c,
Product_Feature__c,
POD_Name__c,
CreatedDate,
ClosedDate,
Case_Type__c,
Age_days__c,
IsEscalated,
CSAT__c,
Internal_Priority__c ,
Case_Owner__c,
RCA__c
from Case WHERE 
Account.Account_Name__c !=null AND
CreatedDate >= LAST_N_DAYS:180
AND ( Type ='Bug Fix' OR Type ='Bug' OR Type ='Change Request' OR Type ='CLOP' OR Type ='Enhancement' OR Type ='Feature Enablement' OR Type ='Feature Request' OR Type ='Incident' OR Type ='Other' OR Type ='Problem' OR Type ='Question' OR Type ='Requirement' OR Type ='Salesforce Access Request' OR Type ='Service Request' OR Type ='Service Scoping' OR Type ='Task' OR Type ='Ticket' )