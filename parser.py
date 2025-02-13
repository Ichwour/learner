import json

def remove_format(filename=r"C:\Users\admin\PycharmProjects\Learner\data\data.json"):
    with open(filename, "r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        for item in data:
            item.pop("format", None)  # Удаляем ключ "format", если он есть

    with open(filename, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

remove_format()