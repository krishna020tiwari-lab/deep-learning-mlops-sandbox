import chromadb
from chromadb.utils import embedding_functions

client = chromadb.PersistentClient(path="./chroma_db")

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

collection = client.get_or_create_collection(name="customer_support_72",embedding_function=embedding_fn,metadata={"hnsw:space": "cosine"})
print(collection.count())
docs_by_category = {
    "billing_refunds": [
        "I would like to get my money back for a faulty order.",
        "Reimbursement process for canceled service charges.",
        "I was charged twice for the same purchase on my card.",
        "How long does a refund take to appear in my bank account?",
        "Please send me a copy of my latest invoice.",
        "The amount on my bill is higher than the price I was quoted.",
        "I want to change the payment method used for my monthly plan.",
        "My credit card was declined but the money left my account.",
        "Can I get a receipt for tax purposes?",
        "There is an unknown transaction on my statement from your company.",
        "Does the promo code discount apply to my renewal payment?",
        "I was billed after cancelling my free trial.",
    ],
    "account": [
        "Can you help me update my account email address?",
        "I forgot my password and cannot log in.",
        "Please delete my account and all of my personal data.",
        "How do I turn on two-factor authentication for my profile?",
        "I want to change the name shown on my profile.",
        "My account has been locked after too many sign-in attempts.",
        "How can I merge two accounts that belong to me?",
        "I want to stop receiving marketing emails from you.",
        "How do I cancel my subscription before the next billing date?",
        "Someone else may have accessed my account without permission.",
        "How do I add another user to my team plan?",
        "Where can I change the phone number saved with my account?",
    ],
    "shipping": [
        "Where can I track the delivery status of my package?",
        "My parcel is delayed and has not arrived yet.",
        "Can I change the delivery address after placing the order?",
        "The tracking page says delivered but I never received anything.",
        "How much does express shipping cost to another country?",
        "Which courier company will deliver my order?",
        "My package arrived damaged at the door.",
        "Can I schedule a delivery for a specific day?",
        "Do you ship to remote areas and islands?",
        "How many days does standard delivery usually take?",
        "The courier left my parcel with a neighbor, how do I find it?",
        "I only received part of my order, the rest is missing.",
    ],
    "tech_support": [
        "The app crashes every time I open it.",
        "The website keeps showing a loading spinner and never finishes.",
        "I cannot connect my device to the Wi-Fi network after the update.",
        "How do I reset the device to factory settings?",
        "The software says my license key is invalid.",
        "Videos stutter and freeze when I stream on my phone.",
        "Bluetooth pairing keeps failing with my headphones.",
        "How can I install the latest firmware update?",
        "The screen flickers and sometimes turns black.",
        "My files are not syncing between my laptop and phone.",
        "I get an error code 504 when I try to upload a document.",
        "The battery drains within a few hours after the update.",
    ],
    "product_info": [
        "What is the difference between the basic and the premium plan?",
        "Does this laptop come with a one year warranty?",
        "Is the jacket available in a larger size?",
        "What colors does the phone come in?",
        "Is this product compatible with my older model?",
        "How much does the monthly subscription cost?",
        "What materials is the backpack made from?",
        "Does the speaker work underwater?",
        "When will the new model be back in stock?",
        "What is included in the box when I buy the camera?",
        "Are there student discounts available?",
        "How much storage does the standard version offer?",
    ],
    "returns_exchanges": [
        "I want to send back the shoes because they do not fit.",
        "How do I exchange my shirt for a different size?",
        "What is your return policy for opened items?",
        "Do I have to pay for return shipping?",
        "Can I return a gift without the original receipt?",
        "How many days do I have to return a product?",
        "I received the wrong item and want to swap it.",
        "Where do I print the return label?",
        "The product is defective and I need a replacement.",
        "Can I return a sale item?",
        "I changed my mind about the purchase, can I send it back?",
        "How do I schedule a pickup for the item I am returning?",
    ],
}

queries_by_category = {
    "billing_refunds": [
        "How do I request a payment refund?",
        "Why did my bank show two deductions for one order?",
        "I need an invoice for my company accounting.",
        "Money was taken from me after I stopped the trial.",
        "How long until the reimbursement reaches my bank?",
    ],
    "account": [
        "I can't sign in because I don't remember my credentials.",
        "Erase everything you know about me.",
        "How do I secure my profile with a second verification step?",
        "Update the mail address linked to my login.",
        "I think a stranger is using my login.",
    ],
    "shipping": [
        "Where is my order right now?",
        "My delivery is late, what is happening?",
        "Can the courier bring it to a different house instead?",
        "How quickly will it arrive if I choose the fastest option?",
        "The box was crushed when it reached me.",
    ],
    "tech_support": [
        "The application closes by itself on startup.",
        "My headphones will not link to the phone.",
        "Everything on the page is stuck loading.",
        "After the upgrade my phone loses charge quickly.",
        "How do I wipe the gadget and start fresh?",
    ],
    "product_info": [
        "Which plan should I pick, standard or premium?",
        "Will this charger work with the previous generation?",
        "Do you sell this in extra large?",
        "What do I get inside the package?",
        "Is there a warranty on this item?",
    ],
    "returns_exchanges": [
        "These boots are too small, how do I send them back?",
        "How can I swap this for a bigger one?",
        "Within how many days can I get my purchase returned?",
        "Do you offer prepaid labels for sending an item back?",
        "The thing I got is broken, I want a new one.",
    ],
}

doc_texts , doc_categories = [], []
for category , texts in docs_by_category.items():
    for text in texts:
        doc_texts.append(text)
        doc_categories.append(category)

query_texts , query_categories = [],[]
for category , texts in queries_by_category.items():
    for text in texts:
        query_texts.append(text)
        query_categories.append(category)

ids = [f"doc_{i}" for i in range(len(doc_texts))]
metadatas = [{"category":c} for c in doc_categories]

collection.upsert(ids=ids,documents=doc_texts,metadatas=metadatas)
print(f"Total indexed documents in collection: {collection.count()}")

eval_results = collection.query(query_texts=query_texts,n_results=5,include=["documents","metadatas","distances"],)
shipping_results = collection.query(query_texts=["where is my order right know"],n_results=3,where={"category":"shipping"},include=["documents","metadatas","distances"])
nomatch_results = collection.query(query_texts=["what is the capital of USA"],n_results=3,include=["documents","metadatas","distances"])

top1_hits = 0
precision_sum = 0.0
for q in range(len(query_texts)):
    wanted = query_categories[q]
    retrieved = [m["category"] for m in eval_results["metadatas"][q]]  # 5 categories
    top1_hits += retrieved[0] == wanted
    precision_sum += retrieved.count(wanted) / 5

print(f"Top-1 accuracy: {top1_hits / len(query_texts):.2%}")
print(f"Precision@5:    {precision_sum / len(query_texts):.2%}")

print("\n --- Shipping Filter ---")
for doc,meta,dist in zip(shipping_results["documents"][0],shipping_results["metadatas"][0],shipping_results["distances"][0]):
    similarity = 1.0 - dist
    print(f"Score Similarity {similarity} | {meta['category']:<10} | {doc}")


print("/ No Match Result")
print(f"Best similarity: {1 - nomatch_results['distances'][0][0]:.4f}")
print(f"Best document: {nomatch_results['documents'][0][0]}")




