import base64
from datetime import datetime
import uuid
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import io

def create_rfq_pdf():
    """Create a realistic RFQ PDF document"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=1*inch)
    
    # Get styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=30,
        textColor=colors.darkblue,
        alignment=1  # Center alignment
    )
    
    story = []
    
    # Title
    title = Paragraph("REQUEST FOR QUOTE (RFQ)", title_style)
    story.append(title)
    story.append(Spacer(1, 20))
    
    # RFQ Details
    rfq_info = [
        ["RFQ Number:", "RFQ-2025-0234"],
        ["Issue Date:", "June 5, 2025"],
        ["Response Due Date:", "June 20, 2025"],
        ["Project:", "Office Equipment Procurement"],
        ["Contact Person:", "Sarah Johnson, Procurement Manager"]
    ]
    
    info_table = Table(rfq_info, colWidths=[2*inch, 3*inch])
    info_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    
    story.append(info_table)
    story.append(Spacer(1, 30))
    
    # Company Information
    company_header = Paragraph("<b>COMPANY INFORMATION</b>", styles['Heading2'])
    story.append(company_header)
    story.append(Spacer(1, 10))
    
    company_text = """
    TechCorp Solutions Inc.<br/>
    1247 Business Park Drive<br/>
    Austin, TX 78759<br/>
    Phone: (512) 555-0123<br/>
    Email: procurement@techcorp.com
    """
    company_para = Paragraph(company_text, styles['Normal'])
    story.append(company_para)
    story.append(Spacer(1, 20))
    
    # Requirements
    req_header = Paragraph("<b>ITEMS REQUESTED</b>", styles['Heading2'])
    story.append(req_header)
    story.append(Spacer(1, 10))
    
    # Items table
    items_data = [
        ["Item #", "Description", "Quantity", "Specifications"],
        ["001", "Office Desk Chairs", "25", "Ergonomic, adjustable height, lumbar support"],
        ["002", "Standing Desks", "15", "Electric height adjustment, 48\"x30\" surface"],
        ["003", "Monitor Arms", "40", "Dual monitor support, VESA compatible"],
        ["004", "Wireless Keyboards", "30", "Bluetooth, ergonomic design"],
        ["005", "Wireless Mice", "30", "Optical, USB receiver included"]
    ]
    
    items_table = Table(items_data, colWidths=[0.8*inch, 2.5*inch, 1*inch, 2.2*inch])
    items_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    
    story.append(items_table)
    story.append(Spacer(1, 20))
    
    # Terms and Conditions
    terms_header = Paragraph("<b>TERMS AND CONDITIONS</b>", styles['Heading2'])
    story.append(terms_header)
    story.append(Spacer(1, 10))
    
    terms_text = """
    1. Quote must be valid for 30 days from submission date<br/>
    2. Payment terms: Net 30 days<br/>
    3. Delivery required within 2 weeks of order placement<br/>
    4. Installation and setup services may be required<br/>
    5. Warranty period: Minimum 1 year manufacturer warranty<br/>
    6. Please include shipping costs in your quote<br/>
    7. Submit quotes in PDF format via email
    """
    terms_para = Paragraph(terms_text, styles['Normal'])
    story.append(terms_para)
    story.append(Spacer(1, 20))
    
    # Footer
    footer_text = """
    <b>SUBMISSION INSTRUCTIONS:</b><br/>
    Please submit your complete quote to: procurement@techcorp.com<br/>
    Subject Line: "RFQ Response - RFQ-2025-0234"<br/>
    Questions: Contact Sarah Johnson at (512) 555-0123 ext. 245
    """
    footer_para = Paragraph(footer_text, styles['Normal'])
    story.append(footer_para)
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

def create_llm_friendly_eml():
    """Create an LLM-friendly EML file by manually constructing the content"""
    
    # Generate boundary
    boundary = f"==============={uuid.uuid4().hex[:16]}=="
    
    # Email headers
    current_time = datetime.now()
    
    # Email body content (plain text, no encoding)
    email_body = """Dear Vendor Partner,

I hope this email finds you well. We are reaching out to request a quote for office equipment as part of our upcoming office expansion project.

TechCorp Solutions Inc. is planning to furnish our new Austin office space and we're seeking competitive quotes from qualified suppliers. We have been impressed with your company's reputation and would like to invite you to participate in this procurement opportunity.

PROJECT DETAILS:
- Project Name: Austin Office Equipment Procurement
- RFQ Number: RFQ-2025-0234
- Estimated Budget: $75,000 - $100,000
- Timeline: Equipment needed by July 1, 2025

WHAT'S INCLUDED:
Please find the detailed Request for Quote (RFQ) document attached to this email. The RFQ contains:
• Complete list of required items and quantities
• Technical specifications for each item
• Delivery requirements and timeline
• Terms and conditions
• Submission guidelines

NEXT STEPS:
1. Review the attached RFQ document carefully
2. Prepare your comprehensive quote including all requested items
3. Submit your response by June 20, 2025, 5:00 PM CST
4. Feel free to contact me with any questions or clarifications needed

We value long-term partnerships and look forward to the possibility of working with your company. Your prompt response would be greatly appreciated.

If you have any questions about this RFQ or need additional information, please don't hesitate to contact me directly at (512) 555-0123 ext. 245 or via email.

Thank you for your time and consideration. We look forward to receiving your competitive quote.

Best regards,

Sarah Johnson
Procurement Manager
TechCorp Solutions Inc.
1247 Business Park Drive
Austin, TX 78759
Phone: (512) 555-0123 ext. 245
Email: sarah.johnson@techcorp.com
Website: www.techcorp.com

---
CONFIDENTIALITY NOTICE: This email and any attachments are confidential and may be privileged. If you are not the intended recipient, please notify the sender immediately and delete this email. Any unauthorized review, use, disclosure, or distribution is prohibited."""

    # PDF content summary for LLM parsing
    pdf_summary = """
=== ATTACHMENT: RFQ-2025-0234_Office_Equipment.pdf ===
Document Type: Request for Quote (RFQ)
RFQ Number: RFQ-2025-0234
Issue Date: June 5, 2025
Response Due Date: June 20, 2025
Project Name: Office Equipment Procurement

COMPANY DETAILS:
Company: TechCorp Solutions Inc.
Address: 1247 Business Park Drive, Austin, TX 78759
Phone: (512) 555-0123
Email: procurement@techcorp.com
Contact Person: Sarah Johnson, Procurement Manager
Extension: 245

PROJECT INFORMATION:
Budget Range: $75,000 - $100,000
Equipment Delivery Required: July 1, 2025
Payment Terms: Net 30 days
Quote Validity: 30 days from submission
Warranty Requirement: Minimum 1 year

ITEMS REQUESTED:
Item 001: Office Desk Chairs - Quantity: 25 units
  Specifications: Ergonomic, adjustable height, lumbar support
  
Item 002: Standing Desks - Quantity: 15 units
  Specifications: Electric height adjustment, 48"x30" surface
  
Item 003: Monitor Arms - Quantity: 40 units
  Specifications: Dual monitor support, VESA compatible
  
Item 004: Wireless Keyboards - Quantity: 30 units
  Specifications: Bluetooth, ergonomic design
  
Item 005: Wireless Mice - Quantity: 30 units
  Specifications: Optical, USB receiver included

IMPORTANT DATES:
- RFQ Issue Date: June 5, 2025
- Quote Submission Deadline: June 20, 2025, 5:00 PM CST
- Equipment Delivery Required: July 1, 2025

SUBMISSION REQUIREMENTS:
- Email submissions to: procurement@techcorp.com
- Subject Line: "RFQ Response - RFQ-2025-0234"
- Include shipping costs in quote
- PDF format preferred for responses

CONTACT INFORMATION:
Primary Contact: Sarah Johnson
Title: Procurement Manager
Phone: (512) 555-0123 ext. 245
Email: sarah.johnson@techcorp.com
Company Website: www.techcorp.com
=== END ATTACHMENT SUMMARY ==="""

    # Create PDF attachment (base64 encoded binary)
    pdf_content = create_rfq_pdf()
    pdf_base64 = base64.b64encode(pdf_content).decode('ascii')
    
    # Construct EML content manually
    eml_content = f"""Content-Type: multipart/mixed; boundary="{boundary}"
MIME-Version: 1.0
From: sarah.johnson@techcorp.com
To: vendor@suppliercompany.com
Cc: procurement-team@techcorp.com
Subject: Request for Quote - Office Equipment Procurement (RFQ-2025-0234)
Date: {current_time.strftime('%a, %d %b %Y %H:%M:%S %z')}
Message-ID: <20250605120000.12345@techcorp.com>

--{boundary}
Content-Type: text/plain; charset=utf-8
Content-Transfer-Encoding: 8bit

{email_body}

--{boundary}
Content-Type: text/plain; charset=utf-8
Content-Transfer-Encoding: 8bit
Content-Description: PDF Content Summary for LLM Processing

{pdf_summary}

--{boundary}
Content-Type: application/pdf
Content-Transfer-Encoding: base64
Content-Disposition: attachment; filename="RFQ-2025-0234_Office_Equipment.pdf"
Content-ID: <rfq_document>

{pdf_base64}

--{boundary}--
"""
    
    # Save to file
    with open('office_equipment_rfq.eml', 'w', encoding='utf-8') as f:
        f.write(eml_content)
    
    print("✅ LLM-optimized EML file created successfully: office_equipment_rfq.eml")
    

if __name__ == "__main__":
    try:
        create_llm_friendly_eml()
    except ImportError as e:
        print("❌ Missing required library. Please install reportlab:")
        print("   pip install reportlab")
        print(f"   Error: {e}")
    except Exception as e:
        print(f"❌ Error creating EML file: {e}")