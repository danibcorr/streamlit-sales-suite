# 3pps


class TestDiscountCalculator:
	"""
	Tests for the discount calculator logic.
	"""

	def test_no_discount(self) -> None:
		"""
		Verifies that a zero discount leaves the price
		unchanged and produces no savings.

		Returns:
			None.
		"""

		price, discount = 100.0, 0
		saving = price * (discount / 100)
		assert saving == 0.0
		assert price - saving == 100.0

	def test_full_discount(self) -> None:
		"""
		Verifies that a 100% discount reduces the price
		to zero and saves the full amount.

		Returns:
			None.
		"""

		price, discount = 50.0, 100
		saving = price * (discount / 100)
		assert saving == 50.0
		assert price - saving == 0.0

	def test_partial_discount(self) -> None:
		"""
		Verifies that a partial discount correctly
		computes the savings and final price.

		Returns:
			None.
		"""

		price, discount = 200.0, 25
		saving = price * (discount / 100)
		assert saving == 50.0
		assert price - saving == 150.0

	def test_zero_price(self) -> None:
		"""
		Verifies that a zero original price results in
		zero savings regardless of the discount.

		Returns:
			None.
		"""

		price, discount = 0.0, 50
		saving = price * (discount / 100)
		assert saving == 0.0
		assert price - saving == 0.0
