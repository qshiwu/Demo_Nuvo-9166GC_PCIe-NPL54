import cv2

def putTextWithBackground(image, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, 
                          font_scale=1, text_color=(255, 255, 255), 
                          bg_color=(0, 0, 0), thickness=2, padding=15):
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Define rectangle coordinates
    x, y = position
    rect_top_left = (x - padding, y - text_height - padding)
    rect_bottom_right = (x + text_width + padding, y + baseline + padding-10)
    
    # Draw filled rectangle
    cv2.rectangle(image, rect_top_left, rect_bottom_right, bg_color, -1)
    
    # Put text on top of rectangle
    cv2.putText(image, text, position, font, font_scale, text_color, thickness)

    return image  # Return modified image

