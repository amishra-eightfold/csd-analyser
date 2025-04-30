# Support Ticket Analysis Exports

This directory contains exported support ticket analysis data organized by customer.

## Directory Structure
```
exports/
├── README.md
├── Customer1/
│   ├── customer1_tickets_20240401_123456_details.csv    # Ticket details
│   ├── customer1_tickets_20240401_123456_history.csv    # Priority history
│   └── customer1_tickets_20240401_123456_emails.csv     # Email messages
└── Customer2/
    └── ...
```

## File Types
- `*_details.csv`: Contains main ticket information including highest priority
- `*_history.csv`: Contains priority change history for tickets
- `*_emails.csv`: Contains email messages related to tickets

## Timestamps
Files are named with timestamps in the format: YYYYMMDD_HHMMSS

## Download
Use the Exports section in the sidebar to download individual files or complete customer data as ZIP archives.
