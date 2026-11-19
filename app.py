"""Flask app: web form (Work 1), REST API ingestion (Work 2), results page."""
from flask import Flask, jsonify, render_template, request

import database
from classifier import process_ticket_full

app = Flask(__name__)
database.init_db()

REQUIRED = ["employee_name", "email", "title", "description"]


def ingest(data):
    """Normalise incoming ticket data into the standard format, classify, store."""
    missing = [f for f in REQUIRED if not str(data.get(f, "")).strip()]
    if missing:
        return None, {"error": f"Missing fields: {', '.join(missing)}"}, 400

    ticket = {
        "employee_name": data["employee_name"].strip(),
        "email": data["email"].strip(),
        "department": str(data.get("department", "")).strip(),
        "title": data["title"].strip(),
        "description": data["description"].strip(),
    }
    # classify on title + description for more context
    text = f"{ticket['title']}. {ticket['description']}"
    result = process_ticket_full(text, data.get("business_impact"))
    ticket.update(result, status="Open")
    ticket["ticket_id"] = database.insert_ticket(ticket)
    return ticket, None, 201


@app.route("/")
def index():
    return render_template("index.html", tickets=database.list_tickets())


@app.route("/ticket", methods=["POST"])
def create_ticket():
    """REST endpoint. Accepts JSON, or form data from the web page."""
    data = request.get_json(silent=True) or request.form.to_dict()
    # the deck's minimal example only sends {"description": ...}
    if "description" in data and not data.get("title"):
        data.setdefault("employee_name", "Unknown")
        data.setdefault("email", "unknown@company.com")
        data["title"] = data["description"][:60]
    ticket, error, code = ingest(data)
    if error:
        return jsonify(error), code
    return jsonify(ticket), code


@app.route("/ticket/<int:ticket_id>", methods=["GET"])
def get_ticket(ticket_id):
    t = database.get_ticket(ticket_id)
    return (jsonify(t), 200) if t else (jsonify({"error": "Not found"}), 404)


@app.route("/tickets", methods=["GET"])
def all_tickets():
    return jsonify(database.list_tickets())


if __name__ == "__main__":
    app.run(debug=True)
