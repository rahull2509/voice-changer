import asyncio
import edge_tts

async def main():
    voices = await edge_tts.list_voices()
    indian = [v for v in voices if "IN" in v.get("Locale", "") or "hi" in v.get("Locale", "").lower()]
    for v in indian:
        print(f"ShortName: {v['ShortName']}, Gender: {v.get('Gender')}, Locale: {v.get('Locale')}, Name: {v.get('FriendlyName')}")

if __name__ == "__main__":
    asyncio.run(main())
