import os
import httpx
import asyncio
from typing import Any, Dict, Optional
import re  # <-- [新增] 导入 re 以进行清理

# Import BaseLLM to use it in function signatures
from guardrails.llms.base import BaseLLM


async def handle_price_query(entities: Dict[str, Any], llm: BaseLLM = None) -> str:
    """Handles user queries about product prices."""

    product_name = entities.get("product", "an unspecified product")
    print(f"--- [Action] Querying price for '{product_name}'... ---")

    # 1. 检查你的"已知"（硬编码）价格
    if "ryzen 9" in product_name.lower():
        return f"The price for {product_name} is $599."
    elif "ryzen 7" in product_name.lower():
        return f"The price for {product_name} is $399."

    # 2. 如果不在已知列表中，则调用 dummyjson API
    else:
        print(
            f"--- [Action] Product not in local list. Querying dummyjson.com API... ---"
        )

        # --- [!! 修复 dummyjson !!] ---
        # 清理搜索词：转为小写，移除标点符号
        # 这使得 API 调用更加健壮
        clean_product_name = re.sub(r"[^\w\s]", "", product_name).lower()
        api_url = "https://dummyjson.com/products/search"
        params = {"q": clean_product_name}
        # --- [!! 结束修复 !!] ---

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(api_url, params=params)
                response.raise_for_status()
                data = response.json()

                if data.get("products") and len(data["products"]) > 0:
                    product = data["products"][0]
                    price = product.get("price")
                    name = product.get("title")
                    return f"According to a search on dummyjson.com, the price for '{name}' is ${price}."
                else:
                    return f"Sorry, I searched online but couldn't find a price for {clean_product_name}."

        except httpx.HTTPStatusError as e:
            print(f"--- [Action] API Error: {e} ---")
            return f"Sorry, I had an error checking the price for {product_name}."
        except Exception as e:
            print(f"--- [Action] General Error: {e} ---")
            return f"Sorry, I ran into an unexpected issue trying to find the price."


# --- [!! 修复 Robot !!] ---
async def handle_robot_arm(entities: Dict[str, Any], llm: BaseLLM = None) -> str:
    """Handles user commands for the robot arm."""
    action = entities.get("action", "do something")

    # --- [!! 修改 !!] ---
    # 使实体提取更健壮，就像我们对航班所做的那样
    target_object = entities.get("target_object", entities.get("object", "something"))
    # --- [!! 结束修改 !!] ---

    print(f"--- [Action] Executing robot command: {action} {target_object}... ---")
    return (
        f"Executing command: The robot arm will now `{action}` the `{target_object}`."
    )


# --- [!! 结束修复 !!] ---


async def handle_flight_query(entities: Dict[str, Any], llm: BaseLLM = None) -> str:
    """
    Handles user queries about flight prices.
    Checks for required slots (origin, dest, trip_type, dates) and asks
    for missing info if not all are provided.
    """

    # (这部分代码是正确的)
    origin = entities.get("origin", entities.get("from"))
    dest = entities.get("destination", entities.get("to"))
    trip_type = entities.get("trip_type")
    departure_date = entities.get("departure_date", entities.get("date"))
    return_date = entities.get("return_date")

    print(f"--- [Action] Handling flight query with entities: {entities} ---")

    if not origin or not dest:
        return "I can help with that. Where are you flying from and to?"

    if not trip_type:
        return f"Sure, for the flight from {origin} to {dest}, are you looking for a one-way or round-trip ticket, and for what dates?"

    if trip_type.lower() == "one-way" and not departure_date:
        return (
            f"Okay, a one-way flight from {origin} to {dest}. For what departure date?"
        )

    if trip_type.lower() == "round-trip":
        if not departure_date or not return_date:
            return f"Okay, a round-trip flight from {origin} to {dest}. What are the departure and return dates?"

    print(f"--- [Action] Simulating API call for a {trip_type} flight... ---")
    await asyncio.sleep(0.75)

    price = 350.00
    if "december" in str(departure_date).lower():
        price = 550.00

    if trip_type.lower() == "round-trip":
        price *= 1.8
        return f"I found a round-trip flight from {origin} to {dest} (Depart: {departure_date}, Return: {return_date}) for ${price:,.2f}."
    else:
        return f"I found a one-way flight from {origin} to {dest} (Depart: {departure_date}) for ${price:,.2f}."
