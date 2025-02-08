from procthor.generation import PROCTHOR10K_ROOM_SPEC_SAMPLER, HouseGenerator


def main() -> int:
    house_generator = HouseGenerator(
        split="train", seed=42, room_spec_sampler=PROCTHOR10K_ROOM_SPEC_SAMPLER
    )

    house, _ = house_generator.sample()
    house.validate(house_generator.controller)

    house.to_json("house.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
