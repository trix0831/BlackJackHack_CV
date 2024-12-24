import CardDetector
import time

def calculate_card_value(card):
    if card in [2, 3, 4, 5, 6]:
        return 1
    elif card in [10, 'J', 'Q', 'K', 'A']:
        return -1
    else:
        return 0

def get_action(player_total, dealer_card, true_count):
    if isinstance(dealer_card, str):
        dealer_card = 10 if dealer_card in ['J', 'Q', 'K'] else 11 if dealer_card == 'A' else int(dealer_card)

    if player_total >= 17:
        return "Stand"
    elif player_total >= 13 and dealer_card <= 6:
        return "Stand"
    elif player_total == 12 and 4 <= dealer_card <= 6:
        return "Stand"
    elif player_total == 11:
        return "Double Down" if true_count >= 2 else "Hit"
    elif player_total == 10:
        return "Double Down" if dealer_card <= 9 and true_count >= 2 else "Hit"
    elif player_total == 9 and 3 <= dealer_card <= 6:
        return "Double Down" if true_count >= 2 else "Hit"
    else:
        return "Hit"

def get_bet_amount(true_count):
    if true_count <= 1:
        return 1
    elif 1 < true_count <= 3:
        return 2
    elif 3 < true_count <= 5:
        return 3
    else:
        return 5

def main():
    running_count = 0
    decks_remaining = 2

    print("Blackjack Assistant (HiLo Strategy)")

    while True:
        print(f"\nCurrent Running Count: {running_count}")
        true_count = running_count / decks_remaining
        print(f"Current True Count: {true_count:.2f}")

        bet_units = get_bet_amount(true_count)
        print(f"Suggested Bet: {bet_units} unit(s)")

        print("\nDetecting cards... Please wait.")
        time.sleep(12)  # Allow card detection to complete

        player_cards = CardDetector.get_detected_player_cards()
        dealer_cards = CardDetector.get_detected_dealer_cards()

        if not player_cards or not dealer_cards:
            print("No cards detected. Try again.")
            continue

        dealer_card = dealer_cards[0]
        print(f"Player Cards: {player_cards}")
        print(f"Dealer Card: {dealer_card}")

        for card in player_cards:
            running_count += calculate_card_value(card)
        running_count += calculate_card_value(dealer_card)

        player_total = sum([10 if card in ['J', 'Q', 'K'] else (11 if card == 'A' else int(card)) for card in player_cards])
        ace_count = player_cards.count('A')
        while player_total > 21 and ace_count > 0:
            player_total -= 10
            ace_count -= 1

        action = get_action(player_total, dealer_card, true_count)
        print(f"Player Total: {player_total}")
        print(f"Recommended Action: {action}")

if __name__ == "__main__":
    main()
