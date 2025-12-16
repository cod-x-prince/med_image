from .models import db, User, Upload, AuditLog
import click
from flask.cli import with_appcontext

def init_db():
    db.create_all()
    # Create default admin user if not exists
    if not User.query.filter_by(username='admin').first():
        from flask_bcrypt import Bcrypt
        bcrypt = Bcrypt()
        hashed_pw = bcrypt.generate_password_hash('admin123').decode('utf-8')
        admin = User(username='admin', password_hash=hashed_pw, role='admin')
        db.session.add(admin)
        db.session.commit()
        print("Initialized database with default admin user.")

@click.command('init-db')
@with_appcontext
def init_db_command():
    """Clear the existing data and create new tables."""
    init_db()
    click.echo('Initialized the database.')

def init_app(app):
    db.init_app(app)
    app.cli.add_command(init_db_command)
