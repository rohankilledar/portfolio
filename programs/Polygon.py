import math

class Polygon:
    def __init__(self,pointA,pointB,pointC,pointD):
        # self.pointA = pointA
        # self.pointB = pointB
        # self.pointC = pointC
        # self.pointD = pointD
        self.points = [pointA,pointB,pointC,pointD]

    def checkPoint(self,pointTup):
        if pointTup in self.points:
            return True
        else:
            return False

    def CP(self,a,b,c):
        ang = math.degrees(
        math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
        return ang + 360 if ang < 0 else ang

    def Convex(self):
        angles = []
        angles.append(self.CP(self.points[0],self.points[1],self.points[2]))
        angles.append(self.CP(self.points[1],self.points[2],self.points[3]))
        angles.append(self.CP(self.points[2],self.points[3],self.points[0]))
        angles.append(self.CP(self.points[3],self.points[0],self.points[1]))
        if 180 in angles:
            return False
        else:
            return True

        


#below code to check if its working
newPoly = Polygon((0,0),(0,1),(1,1),(1,0))
newPoly2 = Polygon((0,0),(0,1),(0.5,0.5),(1,0))
print(newPoly.checkPoint((0,0)))
print(newPoly.checkPoint((2,5)))
print(newPoly.Convex())
print(newPoly2.Convex())