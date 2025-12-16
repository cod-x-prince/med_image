from celery import Celery, Task

def make_celery(app):
    celery = Celery(
        app.import_name,
        backend='cache+memory://',
        broker='memory://',
        include=['med_image.app.tasks'] # Explicitly register tasks
    )
    
    # Use config from app but filter for Celery keys if needed
    # celery.conf.update(app.config) 
    
    # FORCE EAGER MODE for stability (as requested)
    # Use lowercase keys to avoid ImproperlyConfigured error
    celery.conf.update(
        task_always_eager=True,
        task_store_eager_result=True,
        broker_url='memory://',
        result_backend='cache+memory://'
    )
    
    # Set as default app so shared_task uses this configuration
    celery.set_default()
    
    print(f"🔧 Celery Config: Eager Mode ENABLED (Stability Mode)")

    class ContextTask(Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery
