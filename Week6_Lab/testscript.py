import os

# Check if folders exist
train_exists = os.path.exists('seg_train')
test_exists = os.path.exists('seg_test')

print(f"Training folder: {'Found' if train_exists else ' Missing'}")
print(f"Test folder: {'Found' if test_exists else ' Missing'}")

if train_exists:
    categories = sorted(os.listdir('seg_train'))
    print(f"\nCategories: {categories}")
    
    # Count images per category
    for cat in categories:
        count = len(os.listdir(f'seg_train/{cat}'))
        print(f"  {cat}: {count} images")