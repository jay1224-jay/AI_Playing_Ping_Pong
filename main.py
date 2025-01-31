import pygame
from pygame.locals import *
import sys, time, random, math
import pingai
import torch

WIDTH = 600
HEIGHT = 400
pi = 3.1415

pygame.init()
play_surface = pygame.display.set_mode((WIDTH, HEIGHT))
FPS = pygame.time.Clock()
FPS.tick(60)

edge_distance = 60
RECT_WIDTH = 10
RECT_HEIGHT = 70

GAME_FONT = pygame.font.SysFont('Consolas', 30)

WHITE = (255, 255, 255)
GRAY = (150, 150, 150)
RED = (200, 0, 0)

player1_rect = pygame.Rect(edge_distance, HEIGHT/2 - RECT_HEIGHT/2, RECT_WIDTH, RECT_HEIGHT)
player2_rect = pygame.Rect(WIDTH - edge_distance-RECT_WIDTH, HEIGHT/2 - RECT_HEIGHT/2, RECT_WIDTH, RECT_HEIGHT)
#NOTE: player2 is veiwed as AI
ball_x = WIDTH/2
ball_y = HEIGHT/2
angle = random.randint(-60, 60) 
ball_velocity = 0.5
ball_radius = 10

ball_dir = [math.cos(angle*pi/180) * ball_velocity, math.sin(angle*pi/180) * ball_velocity]

#score
player1_score = 0
player2_score = 0

# AI

# Hyperparameters (same as before)
input_size = 5  # Number of input features
hidden_size = 128  # Number of neurons in the hidden layer
output_size = 2  # Number of possible actions

print(" === preparing AI model === ")
device = "cpu"
model = pingai.Model()
model.load_state_dict(torch.load(f'pp_model_{hidden_size}_hiddensize_4_layer_lr.pth'))
model.to(device)
model.eval()
# print(model.state_dict())
print(" === finished AI model === ")

def getDir(dir, player):
    
    if player == 1:
        dir[0] = abs(dir[0])
    else:
        dir[0] = -abs(dir[0])
    
    return dir


while True:
    play_surface.fill((0, 0, 0))
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            # sys.exit()

    # movement
    pressed = pygame.key.get_pressed()
    if pressed[pygame.K_w] and player1_rect.y > 0:
        player1_rect = player1_rect.move(0, -1)
    if pressed[pygame.K_s] and player1_rect.y+RECT_HEIGHT < HEIGHT:
        player1_rect = player1_rect.move(0, 1)
    
    
    # AI
  
    response = abs(model(torch.tensor([[ball_x, ball_y, ball_dir[0], ball_dir[1]]])).item())

    target_y = HEIGHT * response
    # print(response)
    if ( target_y > player2_rect.y ):
        AIResponse = 2
    else:
        AIResponse = 1
    
    # AIResponse = -1
    # player2_rect.y = target_y
    
    if AIResponse == 1 and player2_rect.y > 0:
        player2_rect = player2_rect.move(0, -1)
    elif AIResponse == 2 and player2_rect.y + RECT_HEIGHT < HEIGHT:
        player2_rect = player2_rect.move(0, 1)
    else:
        # print("Error: AI died ~")
        pass
    
    
    """
    if pressed[pygame.K_UP]:
        player2_rect = player2_rect.move(0, -1)
    if pressed[pygame.K_DOWN]:
        player2_rect = player2_rect.move(0, 1)
    """

    ball_x += ball_dir[0]
    ball_y += ball_dir[1]
    

    # collision
    # ball:
    if ball_y-ball_radius <= 0 or ball_y+ball_radius >= HEIGHT:
        ball_dir[1] *= -1
    
    if ball_x-ball_radius <= 0:
        player2_score += 1
        ball_x = WIDTH/2
        ball_y = HEIGHT/2
        angle = random.randint(-50, 50)
        flag = -1 if angle%2 == 0 else 1
        ball_dir = [math.cos(angle * pi / 180) * ball_velocity * flag, math.sin(angle * pi / 180) * ball_velocity]

    elif ball_x + ball_radius >= WIDTH:
        player1_score += 1
        ball_x = WIDTH/2
        ball_y = HEIGHT/2
        angle = random.randint(-50, 50)
        flag = -1 if angle%2 == 0 else 1
        ball_dir = [math.cos(angle * pi / 180) * ball_velocity * flag, math.sin(angle * pi / 180) * ball_velocity]

    # collision with player
    if ball_x + ball_radius >= player2_rect.x and ball_x + ball_radius <= player2_rect.x+3 and \
       ball_y >= player2_rect.y and ball_y <= player2_rect.y + RECT_HEIGHT:
       # ball collided with player2
       ball_dir = getDir(ball_dir, 2)
    
    elif ball_x - ball_radius >= player1_rect.x and ball_x - ball_radius <= player1_rect.x+3 and \
       ball_y >= player1_rect.y and ball_y <= player1_rect.y + RECT_HEIGHT:
       # ball collided with player2
       ball_dir = getDir(ball_dir, 1)

    

    text_surface1 = GAME_FONT.render("Score: %d" % player1_score, 0, WHITE)
    play_surface.blit(text_surface1, (edge_distance, 30))

    text_surface2 = GAME_FONT.render("Score: %d" % player2_score, 0, GRAY)
    play_surface.blit(text_surface2, (WIDTH - edge_distance - 100, 30))
    
    # text_surface3 = GAME_FONT.render("Angle: %d" % angle, 0, WHITE)
    # play_surface.blit(text_surface3, (WIDTH/2-50, 30))

    pygame.draw.rect(play_surface, WHITE, player1_rect)
    pygame.draw.rect(play_surface, GRAY, player2_rect)
    pygame.draw.circle(play_surface, RED, (ball_x, ball_y), ball_radius)

    pygame.display.update()
    time.sleep(0.001)


