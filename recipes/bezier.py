## PYGAME CUBIC BEZIER CURVES DEMO - Copyright David Barker 2009 ##
 
import pygame, sys, math
 
 
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Cubic Bezier curves demo -- Copyright David Barker 2009")
 
 
clock = pygame.time.Clock()
 
points = [[200, 400], [300, 250], [450, 500], [500, 475]]
dragging = False
selected = 0
 
changed = True
 
mousepos = pygame.mouse.get_pos()
clicktolerance = 5
 
charpos = 0.0
 
showdetails = False
showchar = True
 
 
def sqr(num):
    return num*num
 
def cube(num):
    return num*num*num
 
def dist(a, b):
    return math.sqrt(abs((b[0]-a[0])*(b[0]-a[0])) + abs((b[1]-a[1])*(b[1]-a[1])))
 
def vectmult(num, vect):
    # Multiply two vectors
    return [num*vect[0], num*vect[1]]
 
# The following three functions are for summing vectors
 
def twopointsum(a, b):
    return [a[0] + b[0], a[1] + b[1]]
 
def threepointsum(a, b, c):
    return[a[0] + b[0] + c[0], a[1] + b[1] + c[1]]
 
def fourpointsum(a, b, c, d):
    return[a[0] + b[0] + c[0] + d[0], a[1] + b[1] + c[1] + d[1]]
 
def Update():
    global charpos
   
    if dragging:
        # If the user is dragging a point, move the point by the mouse's motion relative to the previous update
        move = pygame.mouse.get_rel()
        points[selected][0] += move[0]
        points[selected][1] += move[1]
 
        # If the seleted point is an endpoint, move the control point it is parent to as well
        if selected == 0:
            points[1][0] += move[0]
            points[1][1] += move[1]
        elif selected == 3:
            points[2][0] += move[0]
            points[2][1] += move[1]
 
    # Move the character along the curve when the user presses the arrow keys
    if pygame.key.get_pressed()[pygame.K_UP] and charpos < 1.0:
        charpos += 0.01
 
    if pygame.key.get_pressed()[pygame.K_DOWN] and charpos > 0.0:
        charpos -= 0.01
 
def Redraw():
    screen.fill((255, 255, 255))
 
    # Draw lines between endpoints and the control points they are parents of
    pygame.draw.aaline(screen, (0, 0, 0), points[0], points[1])
    pygame.draw.aaline(screen, (0, 0, 0), points[2], points[3])
 
    # Draw the Bezier curve
    DrawBezier(showdetails, points)
    changed = False
 
    # Draw circles for the control points
    pygame.draw.circle(screen, (0, 0, 0), points[1], 3, 1)
    pygame.draw.circle(screen, (0, 0, 0), points[2], 3, 1)
 
    # Draw nifty filled circles for the end points
    pygame.draw.circle(screen, (0, 0, 0), points[0], 5)
    pygame.draw.circle(screen, (0, 255, 0), points[0], 5, 1)
    pygame.draw.circle(screen, (0, 0, 0), points[3], 5)
    pygame.draw.circle(screen, (0, 255, 0), points[3], 5, 1)
 
    # Draw the character at its point on the curve
    if showchar:
        fcharpoint = GetBezierPoint(points, charpos)
        charpoint = int(fcharpoint[0]),int(fcharpoint[1])
        pygame.draw.circle(screen, (0, 255, 0), charpoint, 8)
        pygame.draw.circle(screen, (0, 0, 255), charpoint, 8, 2)
 
def DrawBezier(details, points):
    bezierpoints = []
 
    t = 0.0
 
    while t < 1.02:  # Increment through values of t (between 0 and 1)
        # If the user has elected to display the fiddly (and slightly silly) details
        if details:
            q1 = twopointsum(vectmult((1.0 - t), points[0]), vectmult(t, points[1]))
            q2 = twopointsum(vectmult((1.0 - t), points[1]), vectmult(t, points[2]))
            q3 = twopointsum(vectmult((1.0 - t), points[2]), vectmult(t, points[3]))
           
            pygame.draw.aaline(screen, (200, 200, 200), q1, q2)
            pygame.draw.aaline(screen, (200, 200, 200), q2, q3)
 
            r1 = twopointsum(vectmult((1.0 - t), q1), vectmult(t, q2))
            r2 = twopointsum(vectmult((1.0 - t), q2), vectmult(t, q3))
 
            pygame.draw.aaline(screen, (150, 150, 255), r1, r2)
 
       # Append the point on the curve for the current value of t to the list of Bezier points
        bezierpoints.append(GetBezierPoint(points, t))
        t += 0.02  # t++
 
    # Draw the curve
    pygame.draw.aalines(screen, (0, 0, 0), False, bezierpoints)
 
def GetBezierPoint(points, t):
    """EXPLANATION:
The formula for a point on the bezier curve defined by four points (points[0:3]) at a given value of t is as follows:
   B(t) = ((1-t)^3 * points[0]) + (3 * (1-t)^2 * t * points[1]) + (3 * (1-t) * t^2 * points[2]) + (t^3 * points[3])
           (Where 0 <= t <= 1.)
 
Here, the formula has been split into the four component parts, each returning a vactor:
   -part1 = (1-t)^3 * points[0]
   -part2 = 3 * (1-t)^2 * t * points[1]
   -part3 = 3 * (1-t) * t^2 * points[2]
   -part4 = t^3 * points[3]
 
These vectors are then added using the function fourpointsum() to give the final value for B(t).
   """
    part1 = vectmult(cube(1.0 - t), points[0])
    part2 = vectmult((3 * sqr(1.0 - t) * t), points[1])
    part3 = vectmult((3 * (1.0 - t) * sqr(t)), points[2])
    part4 = vectmult(cube(t), points[3])
 
    return fourpointsum(part1, part2, part3, part4)
 
 
while 1:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
 
        elif event.type == pygame.MOUSEMOTION:
            mousepos = pygame.mouse.get_pos()
           
            if dragging:
                changed = True
 
        elif event.type == pygame.MOUSEBUTTONDOWN:
            for i, point in enumerate(points):
                # Test each point in turn to see if the mouse-click was within its click-distance-tolerance radius
                if dist(mousepos, point) < clicktolerance:
                    selected = i
                    dragging = True
                    pygame.mouse.get_rel()
                    break
 
        elif event.type == pygame.MOUSEBUTTONUP:
            dragging = False
 
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_d:  # Toggle details
                showdetails = not showdetails
           
            elif event.key == pygame.K_c:  # Toggle character
                showchar = not showchar
 
    Update()
    Redraw()
   
    if changed:
        pygame.display.update()
 
    clock.tick(30)
 