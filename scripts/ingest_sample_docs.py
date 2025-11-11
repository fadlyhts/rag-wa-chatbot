"""
Sample document ingestion script

Run this script to populate the vector database with sample documents
Usage: python scripts/ingest_sample_docs.py
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.rag.document_processor import document_processor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Sample documents to ingest
SAMPLE_DOCUMENTS = [
    {
        'title': 'Business Hours and Contact Information',
        'text': """
Our Business Hours:
Monday - Friday: 9:00 AM - 6:00 PM (GMT+7)
Saturday: 9:00 AM - 3:00 PM (GMT+7)
Sunday: Closed

Contact Information:
Phone: +62 123 456 7890
Email: support@example.com
Address: Jakarta, Indonesia

We respond to all inquiries within 24 hours during business days.
For urgent matters, please call our hotline.
        """,
        'content_type': 'info',
        'metadata': {'category': 'business_info', 'language': 'en'}
    },
    {
        'title': 'Return and Refund Policy',
        'text': """
Return and Refund Policy:

1. Return Period: You can return products within 30 days of purchase.

2. Conditions for Returns:
   - Product must be in original condition
   - Original packaging must be intact
   - Receipt or proof of purchase required

3. Refund Process:
   - Contact us to initiate a return
   - Ship the item back to our address
   - Refund will be processed within 5-7 business days after we receive the item
   - Refund will be issued to the original payment method

4. Non-Returnable Items:
   - Opened software or digital products
   - Personalized items
   - Clearance items

5. Shipping Costs:
   - Return shipping is free if the product is defective
   - Customer pays return shipping for other reasons

For questions about returns, contact our support team.
        """,
        'content_type': 'policy',
        'metadata': {'category': 'returns', 'language': 'en'}
    },
    {
        'title': 'Product Catalog - Electronics',
        'text': """
Our Electronics Product Catalog:

1. Smartphones:
   - Latest iPhone models: $999 - $1,499
   - Samsung Galaxy series: $799 - $1,299
   - Google Pixel: $699 - $999
   Features: 5G connectivity, advanced cameras, long battery life

2. Laptops:
   - MacBook Pro 14": $1,999
   - MacBook Air: $1,199
   - Dell XPS 15: $1,799
   - Lenovo ThinkPad: $1,299
   Features: High performance, lightweight, excellent displays

3. Tablets:
   - iPad Pro: $1,099
   - Samsung Galaxy Tab: $899
   - Microsoft Surface: $999
   Features: Portable, great for productivity and entertainment

4. Accessories:
   - Wireless earbuds: $99 - $249
   - Phone cases: $19 - $49
   - Screen protectors: $9 - $29
   - Chargers and cables: $19 - $59

All products come with manufacturer warranty. Free shipping on orders over $100.
Volume discounts available for business customers.
        """,
        'content_type': 'product',
        'metadata': {'category': 'electronics', 'language': 'en'}
    },
    {
        'title': 'Frequently Asked Questions (FAQ)',
        'text': """
Frequently Asked Questions:

Q: How long does shipping take?
A: Standard shipping takes 3-5 business days. Express shipping takes 1-2 business days.

Q: Do you ship internationally?
A: Yes, we ship to most countries. Shipping costs vary by location.

Q: What payment methods do you accept?
A: We accept credit cards (Visa, Mastercard, Amex), PayPal, and bank transfers.

Q: Can I track my order?
A: Yes, you will receive a tracking number via email once your order ships.

Q: What if my product is damaged?
A: Contact us immediately with photos. We'll arrange a replacement or refund.

Q: Do you offer warranties?
A: All products come with manufacturer warranty. Extended warranties available for purchase.

Q: Can I cancel my order?
A: Orders can be cancelled within 24 hours of placement. After that, return policies apply.

Q: Do you offer gift wrapping?
A: Yes, gift wrapping is available for $5 per item.

Q: How can I contact customer support?
A: Email us at support@example.com, call +62 123 456 7890, or chat with us on WhatsApp.

Q: Do you have a loyalty program?
A: Yes! Earn points on every purchase. 100 points = $1 discount on future orders.
        """,
        'content_type': 'faq',
        'metadata': {'category': 'faq', 'language': 'en'}
    },
    {
        'title': 'About Our Company',
        'text': """
About Us:

Founded in 2020, we are a leading e-commerce platform specializing in electronics and tech gadgets.
Our mission is to provide high-quality products at competitive prices with excellent customer service.

Company Values:
- Customer First: Your satisfaction is our priority
- Quality Assurance: We only sell authentic, tested products
- Fast Delivery: Quick and reliable shipping
- Transparent Pricing: No hidden fees
- Sustainability: Eco-friendly packaging and practices

Our Team:
We have a dedicated team of 50+ employees including:
- Product specialists
- Customer support representatives
- Logistics experts
- Tech support engineers

Awards and Recognition:
- Best E-commerce Platform 2023
- Customer Service Excellence Award 2022
- Eco-Friendly Business Certification 2021

We serve over 100,000 satisfied customers across Southeast Asia.

Vision: To become the most trusted online electronics retailer in the region.
        """,
        'content_type': 'company',
        'metadata': {'category': 'about', 'language': 'en'}
    },
    {
        'title': 'Technical Support and Troubleshooting',
        'text': """
Technical Support:

Common Issues and Solutions:

1. Device Won't Turn On:
   - Check if battery is charged
   - Try a different charging cable
   - Hold power button for 10+ seconds to force restart
   - Contact support if issue persists

2. Software Issues:
   - Update to latest software version
   - Clear cache and restart device
   - Factory reset as last resort (backup data first)
   - Our tech team can help remotely

3. Connectivity Problems:
   - Restart your router and device
   - Forget and reconnect to WiFi network
   - Check for software updates
   - Reset network settings if needed

4. Performance Issues:
   - Close unused apps
   - Clear storage space (keep 20% free)
   - Disable animations and background processes
   - Consider upgrading if device is old

5. Screen or Display Issues:
   - Adjust brightness settings
   - Check for screen protector interference
   - Test with different apps
   - May need professional repair

Technical Support Hours:
Monday - Friday: 8:00 AM - 8:00 PM
Saturday: 9:00 AM - 5:00 PM
Sunday: 10:00 AM - 4:00 PM

Contact our tech support team for personalized assistance.
Remote support sessions available for complex issues.
        """,
        'content_type': 'support',
        'metadata': {'category': 'technical_support', 'language': 'en'}
    }
]


def main():
    """Main function to ingest sample documents"""
    logger.info("=" * 60)
    logger.info("Starting sample document ingestion")
    logger.info("=" * 60)
    
    # Check if Qdrant is accessible
    try:
        from app.rag.vector_store import vector_store
        if not vector_store.health_check():
            logger.error("Qdrant is not accessible. Please check your configuration.")
            return
        
        logger.info(f"✓ Qdrant connection successful")
        
        # Get collection info
        info = vector_store.get_collection_info()
        logger.info(f"✓ Collection: {info['name']}")
        logger.info(f"  Current vectors: {info.get('vectors_count', 0)}")
        
    except Exception as e:
        logger.error(f"✗ Failed to connect to Qdrant: {e}")
        logger.error("  Please ensure Qdrant is running and OPENAI_API_KEY is set in .env")
        return
    
    # Ingest documents
    logger.info(f"\nIngesting {len(SAMPLE_DOCUMENTS)} sample documents...")
    logger.info("-" * 60)
    
    results = []
    for i, doc in enumerate(SAMPLE_DOCUMENTS, 1):
        logger.info(f"\n[{i}/{len(SAMPLE_DOCUMENTS)}] Processing: {doc['title']}")
        
        try:
            result = document_processor.process_document(
                text=doc['text'],
                title=doc['title'],
                content_type=doc['content_type'],
                metadata=doc.get('metadata', {})
            )
            
            logger.info(f"  ✓ Created {result['chunks_created']} chunks")
            logger.info(f"  ✓ Total tokens: {result['total_tokens']}")
            results.append(result)
            
        except Exception as e:
            logger.error(f"  ✗ Failed to process document: {e}")
            results.append({'error': str(e), 'title': doc['title']})
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Ingestion Summary")
    logger.info("=" * 60)
    
    successful = [r for r in results if 'error' not in r]
    failed = [r for r in results if 'error' in r]
    
    logger.info(f"✓ Successfully processed: {len(successful)}/{len(SAMPLE_DOCUMENTS)} documents")
    logger.info(f"✓ Total chunks created: {sum(r.get('chunks_created', 0) for r in successful)}")
    logger.info(f"✓ Total tokens: {sum(r.get('total_tokens', 0) for r in successful)}")
    
    if failed:
        logger.warning(f"\n✗ Failed: {len(failed)} documents")
        for fail in failed:
            logger.warning(f"  - {fail['title']}: {fail['error']}")
    
    # Final collection status
    try:
        info = vector_store.get_collection_info()
        logger.info(f"\n✓ Final collection status:")
        logger.info(f"  Vectors in collection: {info.get('vectors_count', 0)}")
        logger.info(f"  Status: {info.get('status', 'unknown')}")
    except Exception as e:
        logger.error(f"✗ Could not get collection status: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ Ingestion complete! Your RAG system is ready to use.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
