import cv2
import time
import numpy as np
import os
import Cards
import VideoStream
from collections import defaultdict

# Global constants and shared variables
IM_WIDTH = 1280
IM_HEIGHT = 720
FRAME_RATE = 10
FONT = cv2.FONT_HERSHEY_SIMPLEX
CAMERA_INDEX = 1  # Set your camera index here (0 for default, 1 for external)

train_ranks = None
train_suits = None

# HiLo strategy-related variables
running_count = 0
decks_remaining = 2

# Initialize video stream
videostream = VideoStream.VideoStream((IM_WIDTH, IM_HEIGHT), FRAME_RATE, 2, CAMERA_INDEX).start()
time.sleep(1)

# Load rank and suit training images
path = os.path.dirname(os.path.abspath(__file__))
train_ranks = Cards.load_ranks(path + '/Card_Imgs/')
train_suits = Cards.load_suits(path + '/Card_Imgs/')


def calculate_card_value(card):
    """Calculate HiLo value for a card."""
    if card in [2, 3, 4, 5, 6]:
        return 1
    elif card in [10, 'J', 'Q', 'K', 'A']:
        return -1
    else:
        return 0


def get_action(player_total, dealer_card, true_count):
    """Return the action based on HiLo strategy."""
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
    """Suggest the bet amount in units based on the true count."""
    if true_count <= 1:
        return 1  # Minimum bet
    elif 1 < true_count <= 3:
        return 2  # Moderate bet
    elif 3 < true_count <= 5:
        return 3  # Higher bet
    else:
        return 5  # Maximum bet


def detect_cards(scan_duration=10):
    """Detect cards for a given duration and return detected player and dealer cards."""
    dealer_confidence = defaultdict(list)
    player_confidence = defaultdict(list)
    scan_start_time = time.time()

    while time.time() - scan_start_time < scan_duration:
        image = videostream.read()
        pre_proc = Cards.preprocess_image(image)
        cnts_sort, cnt_is_card = Cards.find_cards(pre_proc)

        if len(cnts_sort) != 0:
            for i in range(len(cnts_sort)):
                if cnt_is_card[i] == 1:
                    card = Cards.preprocess_card(cnts_sort[i], image)
                    card.best_rank_match, _, rank_diff, _ = Cards.match_card(card, train_ranks, train_suits)
                    confidence = 1.0 / (rank_diff + 1)
                    centroid_y = np.mean(card.contour[:, :, 1])

                    if centroid_y < IM_HEIGHT / 2:
                        dealer_confidence[card.best_rank_match].append(confidence)
                    else:
                        player_confidence[card.best_rank_match].append(confidence)

        cv2.putText(image, "Detecting Cards...", (10, 26), FONT, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("Card Detector", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    dealer_cards = get_best_cards(dealer_confidence)
    player_cards = get_best_cards(player_confidence)
    return player_cards, dealer_cards


def get_best_cards(cards_confidence):
    """Get the ranks with the highest confidence scores."""
    rank_map = {
        'Ace': 'A',
        'Two': '2',
        'Three': '3',
        'Four': '4',
        'Five': '5',
        'Six': '6',
        'Seven': '7',
        'Eight': '8',
        'Nine': '9',
        'Ten': '10',
        'Jack': 'J',
        'Queen': 'Q',
        'King': 'K'
    }
    best_cards = []
    for rank, confidences in cards_confidence.items():
        if confidences and rank != "Unknown":
            best_confidence = max(confidences)
            best_cards.append((rank, best_confidence))
    return [rank_map.get(rank, rank) for rank, _ in best_cards]


def main():
    global running_count

    print("Blackjack HiLo Strategy Assistant")
    print("Press 's' to start detecting cards or 'q' to quit.")

    while True:
        image = videostream.read()
        cv2.putText(image, "Press 's' to scan cards or 'q' to quit.", (10, 26), FONT, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("Card Detector", image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            print("\nStarting 10-second scan...")
            player_cards, dealer_cards = detect_cards()

            if not player_cards or not dealer_cards:
                print("No cards detected. Try again.")
                continue

            dealer_card = dealer_cards[0]
            print(f"Player Cards: {player_cards}")
            print(f"Dealer Card: {dealer_card}")

            for card in player_cards:
                running_count += calculate_card_value(card)
            running_count += calculate_card_value(dealer_card)

            true_count = running_count / decks_remaining
            print(f"\nCurrent Running Count: {running_count}")
            print(f"True Count: {true_count:.2f}")

            bet_units = get_bet_amount(true_count)
            print(f"Suggested Bet: {bet_units} unit(s)")

            player_total = sum([10 if card in ['J', 'Q', 'K'] else (11 if card == 'A' else int(card)) for card in player_cards])
            ace_count = player_cards.count('A')
            while player_total > 21 and ace_count > 0:
                player_total -= 10
                ace_count -= 1

            action = get_action(player_total, dealer_card, true_count)
            print(f"Player Total: {player_total}")
            print(f"Recommended Action: {action}")
            print("\nPress 's' to start a new round or 'q' to quit.")

        elif key == ord('q'):
            print("Exiting...")
            break

    cv2.destroyAllWindows()
    videostream.stop()


if __name__ == "__main__":
    main()
