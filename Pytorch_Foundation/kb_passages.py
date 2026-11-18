
PASSAGES_BY_CATEGORY = {
    "billing_refunds": [
        "Refunds: if an order is faulty or not as described, request a refund within 30 days of delivery. "
        "Open the Orders page and choose 'Request refund'. Approved refunds are sent back to the original payment method.",
        "Refund timing: after a refund is approved, it usually appears in your bank account within 5 to 7 business days, "
        "depending on your bank.",
        "Duplicate charges: if you see two charges for one order, one of them is usually a temporary authorization that "
        "disappears within 3 to 5 business days. If it stays longer, contact support with the order number and we will "
        "refund the extra charge.",
        "Invoices and receipts: you can download an invoice or tax receipt for every payment from Account Settings, "
        "under Billing History. Company details such as a tax number can be added before you download.",
        "Free trials: a free trial can be cancelled at any time before it ends, without any charge. If you were billed "
        "after cancelling, contact support and we will refund the payment made after the cancellation date.",
    ],
    "account": [
        "Password reset: choose 'Forgot password' on the sign-in page and enter your email address. We send a reset link "
        "that is valid for 30 minutes.",
        "Delete your account: go to Account Settings, then Privacy, then 'Delete account'. All personal data is erased "
        "within 30 days, and this cannot be undone.",
        "Two-factor authentication: turn it on under Account Settings, then Security. You can use an authenticator app "
        "or text message codes as the second verification step.",
        "Change your email: update the email address under Account Settings, then Profile. We send a confirmation link "
        "to the new address, and the change applies after you click it.",
        "Suspicious activity: if you think someone else is using your account, reset your password right away, sign out "
        "of all devices from the Security settings, and contact support so we can lock the account and review recent activity.",
    ],
    "shipping": [
        "Order tracking: open Orders and select 'Track package' to see the live location and the expected delivery date. "
        "The tracking link is also in your shipping confirmation email.",
        "Delayed deliveries: standard delivery takes 3 to 7 business days. If your parcel is more than 3 days past the "
        "expected date, contact support and we will contact the courier and offer a replacement or a refund.",
        "Changing the address: you can change the delivery address from the Orders page until the order is handed to the "
        "courier. After that, contact the courier to redirect the parcel to a different address.",
        "Delivery speed: express shipping delivers in 1 to 2 business days within the country and in 3 to 5 business days "
        "internationally, for an extra fee that is shown at checkout.",
        "Damaged parcels: if your package arrives damaged, take photos of the box and the item within 48 hours and send "
        "them to support. We will send a free replacement or refund the order.",
    ],
    "tech_support": [
        "App crashes: update the app to the latest version, restart your phone, and clear the app cache. If it still "
        "closes on startup, uninstall and reinstall the app.",
        "Bluetooth problems: remove the device from the Bluetooth list, turn Bluetooth off and on, put the headphones in "
        "pairing mode, and pair them again.",
        "Page not loading: refresh the page, clear your browser cache, or try a different browser or a private window. "
        "Also check that your internet connection is stable.",
        "Battery drain after an update: restart the device, check which apps use the most battery in the settings, and "
        "turn off background refresh for apps you rarely use. Install the latest update, because it often includes battery fixes.",
        "Factory reset: back up your data first, then open Settings, choose 'Reset', and select 'Factory reset'. The "
        "device erases everything and restarts as new.",
    ],
    "product_info": [
        "Plans: the Standard plan includes the core features for one user. The Premium plan adds priority support, more "
        "storage and up to five users. You can switch plans at any time.",
        "Compatibility: our chargers and accessories work with the current generation and with the previous two "
        "generations of our devices. The product page lists every compatible model.",
        "Sizes: clothing is sold from extra small to extra large. Sizes that are sold out show as unavailable on the "
        "product page, and you can use 'Notify me' to get an email when they return.",
        "In the box: every device ships with the device itself, a charging cable, a quick-start guide and a warranty "
        "card. A power adapter is sold separately.",
        "Warranty: most products include a one year limited warranty that covers manufacturing defects but not accidental "
        "damage. Registering the product extends the warranty by six months.",
    ],
    "returns_exchanges": [
        "Return window: you can return most items within 30 days of delivery if they are unused and in the original "
        "packaging. Sale items can be returned within 14 days.",
        "How to return an item: open Orders, choose 'Return item' and select a reason. We email you a prepaid return "
        "label that you can print, or you can show the QR code at a drop-off point.",
        "Exchanges: to swap an item for a different size or color, choose 'Exchange' in Orders. We ship the new item as "
        "soon as the courier scans your returned parcel.",
        "Wrong or defective items: if you receive the wrong item or a broken one, report it within 14 days with a photo. "
        "We send a replacement at no cost and arrange a free pickup of the faulty item.",
        "Return shipping costs: return shipping is free when the item is faulty or wrong. For other returns, a flat fee "
        "of 4 dollars is deducted from the refund unless you use a drop-off point.",
    ],
}

# Questions the knowledge base CAN answer (question, expected category of the best passage)
ANSWERABLE_TESTS = [
    ("How do I request a payment refund?", "billing_refunds"),
    ("How long until the reimbursement reaches my bank?", "billing_refunds"),
    ("I need an invoice for my company accounting.", "billing_refunds"),
    ("Erase everything you know about me.", "account"),
    ("I think a stranger is using my login.", "account"),
    ("Where is my order right now?", "shipping"),
    ("The box was crushed when it reached me.", "shipping"),
    ("My headphones will not link to the phone.", "tech_support"),
    ("How do I wipe the gadget and start fresh?", "tech_support"),
    ("Which plan should I pick, standard or premium?", "product_info"),
    ("What do I get inside the package?", "product_info"),
    ("Do you offer prepaid labels for sending an item back?", "returns_exchanges"),
    ("The thing I got is broken, I want a new one.", "returns_exchanges"),
    ("Within how many days can I get my purchase returned?", "returns_exchanges"),
]

# Questions the knowledge base can NOT answer: the pipeline should say "I don't know"
NOT_IN_KB = [
    "What is the capital of France?",
    "How do I bake a chocolate cake?",
    "Who won the football world cup?",
    "Do you price match competitors?",
    "Do you have a store in Mumbai?",
    "What are your customer support phone hours on Sundays?",
]

# Tricky: use domain words, mix topics, or sound answerable without being answerable
TRICKY_TESTS = [
    "Can I get a refund on a sale item after 20 days?",
    "My parcel arrived damaged and the item is out of warranty, what happens?",
    "Can I use the premium plan with six users?",
    "Do you ship to the moon?",
]

# ---- Flatten into parallel lists ----
PASSAGES = []
PASSAGE_CATEGORIES = []
for category, texts in PASSAGES_BY_CATEGORY.items():
    for text in texts:
        PASSAGES.append(text)
        PASSAGE_CATEGORIES.append(category)
