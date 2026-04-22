# Hinge Task Profiles

Create one folder per profile under this directory:

`bci/profiles/<profile_name>/`

Each profile folder should contain:

- `picture_1.jpg`
- `picture_2.jpg`
- `picture_3.jpg`
- `profile_metadata.json`

`profile_metadata.json` should include:

```json
{
  "name": "Alex",
  "occupation": "Designer",
  "age": 28,
  "prompt_1": {
    "prompt": "Typical Sunday",
    "response": "Coffee, a hike, and a good book."
  },
  "prompt_2": {
    "prompt": "Green flags I look for",
    "response": "Kindness, curiosity, and good banter."
  }
}
```
