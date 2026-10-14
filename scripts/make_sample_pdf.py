"""Regenerate sample_docs/information_security_policy.pdf (requires `pip install reportlab`)."""

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

OUTPUT = Path(__file__).resolve().parents[1] / "sample_docs" / "information_security_policy.pdf"

PAGES = [
    [
        ("title", "Nimbus Analytics Information Security Policy"),
        ("body", "Version 2.1, approved by the Chief Information Security Officer on 10 February 2025. "
                 "This policy applies to all employees, contractors and interns of Nimbus Analytics."),
        ("h", "1. Data Classification"),
        ("body", "All information is classified into one of four levels: Public, Internal, Confidential and "
                 "Restricted. Customer data, credentials and encryption keys are always classified as Restricted. "
                 "Restricted data must never be stored on personal devices or shared through personal email or "
                 "messaging accounts."),
        ("h", "2. Passwords and Authentication"),
        ("body", "Passwords for company accounts must be at least 14 characters long. Multi-factor authentication "
                 "(MFA) is mandatory for every company system that supports it. All employees must store "
                 "passwords in the company-provided 1Password vault. Passwords are not rotated on a fixed "
                 "schedule; they must be changed immediately if a compromise is suspected."),
    ],
    [
        ("h", "3. Remote Access"),
        ("body", "Internal systems may only be accessed from outside an office through the company VPN, "
                 "NimbusConnect. Public Wi-Fi networks, such as those in cafes, airports and hotels, may be used "
                 "only while NimbusConnect is active. Screens must lock automatically after 5 minutes of "
                 "inactivity."),
        ("h", "4. Device Security"),
        ("body", "Full-disk encryption is mandatory on all laptops and mobile devices that access company data. "
                 "Operating system security updates must be installed within 7 days of release."),
        ("body", "Lost or stolen devices must be reported to the Security team within 4 hours of discovering the "
                 "loss, by email to security@nimbusanalytics.example or through the 24/7 security hotline. The "
                 "Security team will remotely lock and wipe the device. Delayed reporting is itself treated as a "
                 "security incident."),
    ],
    [
        ("h", "5. Incident Response"),
        ("body", "Security incidents are classified by severity. A Severity 1 incident is a confirmed breach of "
                 "customer data. For a Severity 1 incident an incident commander must be appointed within 30 "
                 "minutes, and affected customers must be notified within 72 hours of the breach being confirmed. "
                 "Regulators are notified as required by applicable law."),
        ("h", "6. Access Management"),
        ("body", "Access rights to production systems are reviewed every quarter by system owners. When an "
                 "employee leaves the company, all of their access must be revoked within 24 hours of their last "
                 "working day."),
        ("h", "7. Encryption"),
        ("body", "Data at rest is encrypted with AES-256. Data in transit must be protected with TLS 1.2 or "
                 "higher."),
        ("h", "8. Security Awareness"),
        ("body", "All employees complete security awareness training when they join and then once every year. "
                 "Simulated phishing exercises are run every month."),
    ],
]


def main() -> None:
    styles = getSampleStyleSheet()
    style = {"title": styles["Title"], "h": styles["Heading2"], "body": styles["BodyText"]}
    story = []
    for index, page in enumerate(PAGES):
        for kind, text in page:
            story.append(Paragraph(text, style[kind]))
            story.append(Spacer(1, 6))
        if index < len(PAGES) - 1:
            story.append(PageBreak())
    SimpleDocTemplate(str(OUTPUT), pagesize=A4, title="Information Security Policy").build(story)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
