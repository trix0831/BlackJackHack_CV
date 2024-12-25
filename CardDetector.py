import cv2
import numpy as np
import time
import os
import Cards
import VideoStream
from collections import defaultdict

# Shared variables for player and dealer cards
detected_player_cards = []
detected_dealer_cards = []

def get_detected_player_cards():
    global detected_player_cards
    return detected_player_cards

def get_detected_dealer_cards():
    global detected_dealer_cards
    return detected_dealer_cards

def reset_detected_cards():
    global detected_player_cards, detected_dealer_cards
    detected_player_cards = []
    detected_dealer_cards = []

def convert_rank(rank):
    """Convert card rank to simplified format."""
    rank_map = {
        'Ace': 'A', 'Two': '2', 'Three': '3', 'Four': '4',
        'Five': '5', 'Six': '6', 'Seven': '7', 'Eight': '8',
        'Nine': '9', 'Ten': '10', 'Jack': 'J', 'Queen': 'Q', 'King': 'K'
    }
    return rank_map.get(rank, rank)

def get_best_cards(cards_confidence):
    """Get ranks with the highest confidence scores."""
    best_cards = [
        (convert_rank(rank), max(confidences))
        for rank, confidences in cards_confidence.items()
        if confidences and rank != "Unknown"
    ]
    return [rank for rank, _ in best_cards]

def main():
    # Camera and frame settings
    IM_WIDTH = 1280
    IM_HEIGHT = 720
    FRAME_RATE = 10
    CAMERA_INDEX = 1  # Set your camera index here (0 for default camera, 1 for external camera)

    # Variables for timing and font
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Initialize the video stream
    videostream = VideoStream.VideoStream((IM_WIDTH, IM_HEIGHT), FRAME_RATE, 2, CAMERA_INDEX).start()
    time.sleep(1)

    # Load card rank and suit images
    path = os.path.dirname(os.path.abspath(__file__))
    train_ranks = Cards.load_ranks(path + '/Card_Imgs/')
    train_suits = Cards.load_suits(path + '/Card_Imgs/')

    # Scanning and card detection variables
    cam_quit = False
    is_scanning = False
    scan_start_time = 0
    scan_duration = 10  # seconds

    global detected_player_cards, detected_dealer_cards
    dealer_confidence = defaultdict(list)
    player_confidence = defaultdict(list)

    print("Press 's' to start a scan or 'q' to quit")

    while not cam_quit:
        image = videostream.read()
        t1 = cv2.getTickCount()

        if is_scanning:
            elapsed_time = time.time() - scan_start_time

            if elapsed_time >= scan_duration:
                # Scanning complete
                is_scanning = False
                detected_dealer_cards = get_best_cards(dealer_confidence)
                detected_player_cards = get_best_cards(player_confidence)

                print("\nDealer's cards:", ' '.join(sorted(detected_dealer_cards)))
                print("Player's cards:", ' '.join(sorted(detected_player_cards)))

                dealer_confidence.clear()
                player_confidence.clear()
                print("\nPress 's' to start another scan or 'q' to quit")
            else:
                # Preprocess image and detect cards
                pre_proc = Cards.preprocess_image(image)
                cnts_sort, cnt_is_card = Cards.find_cards(pre_proc)

                if cnts_sort:
                    for i, is_card in enumerate(cnt_is_card):
                        if is_card:
                            card = Cards.preprocess_card(cnts_sort[i], image)
                            card.best_rank_match, _, rank_diff, _ = Cards.match_card(card, train_ranks, train_suits)
                            confidence = 1.0 / (rank_diff + 1)
                            centroid_y = np.mean(card.contour[:, :, 1])

                            # Assign confidence to dealer or player based on card position
                            if centroid_y < IM_HEIGHT / 2:
                                dealer_confidence[card.best_rank_match].append(confidence)
                            else:
                                player_confidence[card.best_rank_match].append(confidence)
                            image = Cards.draw_results(image, card)

                remaining_time = int(scan_duration - elapsed_time)
                cv2.putText(image, f"Scanning... {remaining_time}s", (10, 26), font, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(image, "Press 's' to start scanning", (10, 26), font, 0.7, (255, 0, 255), 2, cv2.LINE_AA)

        # Display the processed image
        cv2.imshow("Card Detector", image)
        t2 = cv2.getTickCount()
        frame_rate_calc = 1 / ((t2 - t1) / freq)

        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s") and not is_scanning:
            is_scanning = True
            scan_start_time = time.time()
            reset_detected_cards()
            print("\nStarting scan...")
        elif key == ord("q"):
            cam_quit = True

    # Cleanup
    cv2.destroyAllWindows()
    videostream.stop()

if __name__ == "__main__":
    main()
