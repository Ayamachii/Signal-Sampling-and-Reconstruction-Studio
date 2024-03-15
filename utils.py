import random


def generate_random_color() -> str:
    """
    Generate a random color code.
    """
    color = "#{:02x}{:02x}{:02x}".format(
        random.randint(0, 255), random.randint(100, 255), random.randint(100, 255)
    )
    return color