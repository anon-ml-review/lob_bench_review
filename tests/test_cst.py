import jax
import jax.numpy as jnp
import cst
from cst import CSTParams


def test_cst_step_scan():
    params = CSTParams()
    book = cst.init_book(
        jnp.array([100_0100, 1, 100_0000, 2, 101_1000, 3, 98_0000, 4, 0, 0, 0, 0]),
        params
    )
    print("INITIAL BOOK")
    print(book)
    print(cst.get_l2_book(book, params, 5))

    co_theta = jnp.ones(params.num_ticks) * 0.1
    base_rates = cst.get_event_base_rates(params, co_theta)
    print("base_rates", base_rates)

    rng = jax.random.PRNGKey(0)

    c, (msgs, l2_books) = jax.lax.scan(
        cst.make_step_book_scannable(5),
        (book, base_rates, params, rng),
        length=100,
    )
    print(msgs)
    print(l2_books)


def test_cst_step_for():
    params = CSTParams()
    book = cst.init_book(
        jnp.array([100_0100, 1, 100_0000, 2, 101_1000, 3, 98_0000, 4, 0, 0, 0, 0]),
        params
    )
    print("INITIAL BOOK")
    print(book)
    print(cst.get_l2_book(book, params, 5))

    co_theta = jnp.ones(params.num_ticks) * 0.1
    base_rates = cst.get_event_base_rates(params, co_theta)
    print("base_rates", base_rates)

    rng = jax.random.PRNGKey(0)

    for i in range(100):
        book, message, rng = cst.step_book(book, base_rates, params, rng)
        print(message)
        print("BOOK AFTER STEP", i)
        print(cst.get_l2_book(book, params, 5))
        # print(book)
        print()
