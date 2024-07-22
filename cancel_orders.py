import asyncio
import planet
from coastseg_planet import download


# get the order ids
async def cancel_order_by_name(
    order_name: str,
    order_states: list = None,
    **kwargs,
):
    async with planet.Session() as sess:
        cl = sess.client("orders")
        # check if an existing order with the same name exists
        order_ids = await download.get_order_ids_by_name(
            cl, order_name, states=order_states
        )
        print(f"order_ids: {order_ids}")
        canceled_orders_info=await cl.cancel_orders(order_ids)
        print(f"canceled_orders_info: {canceled_orders_info}")


order_name = "Santa_Cruz_boardwalk_TOAR_enabled_analytic_udm2_full_dataset_cloud_cover_60"

# cancel all the order ids
asyncio.run(cancel_order_by_name(order_name,order_states=["queued","running"]))