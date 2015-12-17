//=============================================================================================
// Szamitogepes grafika hazi feladat keret. Ervenyes 2014-tol.          
// A //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// sorokon beluli reszben celszeru garazdalkodni, mert a tobbit ugyis toroljuk. 
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat. 
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni (printf is fajlmuvelet!)
// - new operatort hivni az onInitialization függvényt kivéve, a lefoglalt adat korrekt felszabadítása nélkül 
// - felesleges programsorokat a beadott programban hagyni
// - tovabbi kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan gl/glu/glut fuggvenyek hasznalhatok, amelyek
// 1. Az oran a feladatkiadasig elhangzottak ES (logikai AND muvelet)
// 2. Az alabbi listaban szerepelnek:  
// Rendering pass: glBegin, glVertex[2|3]f, glColor3f, glNormal3f, glTexCoord2f, glEnd, glDrawPixels
// Transzformaciok: glViewport, glMatrixMode, glLoadIdentity, glMultMatrixf, gluOrtho2D, 
// glTranslatef, glRotatef, glScalef, gluLookAt, gluPerspective, glPushMatrix, glPopMatrix,
// Illuminacio: glMaterialfv, glMaterialfv, glMaterialf, glLightfv
// Texturazas: glGenTextures, glBindTexture, glTexParameteri, glTexImage2D, glTexEnvi, 
// Pipeline vezerles: glShadeModel, glEnable/Disable a kovetkezokre:
// GL_LIGHTING, GL_NORMALIZE, GL_DEPTH_TEST, GL_CULL_FACE, GL_TEXTURE_2D, GL_BLEND, GL_LIGHT[0..7]
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    :
// Neptun :
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy 
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem. 
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a 
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb 
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem, 
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.  
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat 
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================

#define _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>

#if defined(__APPLE__)                                                                                                                                                                                                            
#include <OpenGL/gl.h>                                                                                                                                                                                                            
#include <OpenGL/glu.h>                                                                                                                                                                                                           
#include <GLUT/glut.h>                                                                                                                                                                                                            
#else                                                                                                                                                                                                                             
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)                                                                                                                                                                       
#include <windows.h>                                                                                                                                                                                                              
#endif                                                                                                                                                                                                                            
#include <GL/gl.h>                                                                                                                                                                                                                
#include <GL/glu.h>                                                                                                                                                                                                               
#include <GL/glut.h>                                                                                                                                                                                                              
#endif          


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Innentol modosithatod...

#define PI 3.14159

struct Vector {
   float x, y, z;

   Vector( ) { 
	x = y = z = 0;
   }
   Vector(float x0, float y0, float z0 = 0) { 
	x = x0; y = y0; z = z0;
   }
   Vector operator*(float a) { 
	return Vector(x * a, y * a, z * a); 
   }
   Vector operator+(const Vector& v) {
 	return Vector(x + v.x, y + v.y, z + v.z); 
   }
   Vector operator-(const Vector& v) {
 	return Vector(x - v.x, y - v.y, z - v.z); 
   }
   float operator*(const Vector& v) {
	return (x * v.x + y * v.y + z * v.z); 
   }
   Vector operator%(const Vector& v) {
	return Vector(y*v.z-z*v.y, z*v.x - x*v.z, x*v.y - y*v.x);
   }
   float Length() { return sqrt(x * x + y * y + z * z); }
   Vector operator/(const float l) {
	   return Vector(x / l, y / l, z / l);
   }
   Vector normalize() {
	   return (*this) / (*this).Length();
   }
};
 
struct Color {
   float r, g, b;

   Color( ) { 
	r = g = b = 0;
   }
   Color(float r0, float g0, float b0) { 
	r = r0; g = g0; b = b0;
   }
   Color operator*(float a) { 
	return Color(r * a, g * a, b * a); 
   }
   Color operator*(const Color& c) { 
	return Color(r * c.r, g * c.g, b * c.b); 
   }
   Color operator+(const Color& c) {
 	return Color(r + c.r, g + c.g, b + c.b); 
   }
   Color operator+=(const Color& c) {
	   *this = *this + c;
	   return *this;
   }
   Color operator-(const Color& c) {
	   return Color(r - c.r, g - c.g, b - c.b);
   }
};

const int screenWidth = 600;
const int screenHeight = 600;


Color image[screenWidth*screenHeight];

class Material {

	bool reflective;
	bool refractive;

public:
	
	Material(bool refl, bool refr) : reflective(refl), refractive(refr) {

	}

	bool isReflective() {
		return reflective;
	}

	bool isRefractive() {
		return refractive;
	}

	virtual Vector reflect(Vector v, Vector normal) {
		return Vector();
	}

	virtual Vector refract(Vector v, Vector normal) {
		return Vector();
	}

	virtual Color shade(Vector v, Vector normal, Vector lightDir, Color inRad, float textureDist) = 0;

	virtual Color fresnel(Vector v, Vector normal) {
		return Color();
	}

	virtual Color getColor(float textureDist) {
		return Color();
	}

};

class SmoothMaterial : public Material {
	Color F0;
	float n;

public:
	SmoothMaterial(float n1, Color F01, bool refl, bool refr) : Material(refl, refr), n(n1), F0(F01) {
		
	}

	Vector reflect(Vector v, Vector normal) {
		float cosa = normal*v;
		return v - normal*cosa*2.0f;
	}

	Vector refract(Vector v, Vector normal) {
		float ior = n;
		float cosa = -1 * (normal*v);
		if (cosa < 0) {
			cosa = -1 * cosa;
			normal = normal * (-1);
			ior = 1 / ior;
		}
		float disc = 1 - (1 - pow(cosa, 2)) / pow(ior, 2);
		if (disc < 0) {
			return reflect(v, normal);
		}
		return v / ior + normal*(cosa / ior - sqrt(disc));
	}

	Color shade(Vector v, Vector normal, Vector lightDir, Color inRad, float textureDist) {
		return Color(0.0, 0.0, 0.0);
	}

	Color fresnel(Vector v, Vector normal) {
		float tetha1 = acosf(v*normal);
		return F0 + ((Color(1.0f, 1.0f, 1.f) - F0)*pow((1.0f - cos(tetha1)), 5));
	}
};

class RoughMaterial : public Material{
	Color kd;
	Color ks;
	float shininess;

public:
	RoughMaterial(Color kd1, Color ks1, float sh) : Material(false, false), kd(kd1), ks(ks1), shininess(sh) {

	}

	Color shade(Vector v, Vector normal, Vector lightDir, Color inRad, float textureDist) {
		Color reflRad(0, 0, 0);
		float cosTetha = lightDir*normal;
		if (cosTetha < 0) {
			return reflRad;
		}
		if (((5.5 < textureDist) && (textureDist < 6.0)) || ((6.5 < textureDist) && (textureDist < 7.0)) || ((7.5 < textureDist) && (textureDist < 8.0)))
		{
			reflRad = inRad*(kd*0.5)*cosTetha;
		}
		else {
			reflRad = inRad*kd*cosTetha;
		}
		
		Vector halfway = (lightDir - v).normalize();
		Vector normalizedNormal = normal.normalize();
		float cosDelta = normalizedNormal*halfway;
		if (cosDelta < 0) {
			return reflRad;
		}
		return reflRad + inRad*ks*pow(cosDelta, shininess);
	}

	Color getColor(float textureDist) {
		if (((5.5 < textureDist) && (textureDist < 6.0)) || ((6.5 < textureDist) && (textureDist < 7.0)) || ((7.5 < textureDist) && (textureDist < 8.0))) {
			return kd*0.5;
		}
		return kd;
	}
};

struct Hit {
	float t;
	Vector position;
	Vector normal;
	Material* material;
	Hit() { t = -1; }
};

struct Ray {
	float prevTime;
	Vector eye;
	Vector p;
	Vector dir;
	Ray() {

	}
	Ray(Vector x, Vector d, bool normalize, float ellapsed) : prevTime(ellapsed), p(x), eye(x) {
		if (normalize) {
			dir = (d - x).normalize();
		}
		else {
			dir = d;
		}
	}
};

enum Axis {
	X, Y, Z
};

struct RotationMatrix {
	float r00, r01, r02;
	float r10, r11, r12;
	float r20, r21, r22;

	RotationMatrix() {

	}

	RotationMatrix(float angle, Axis ax) {
		float a = angle / 180 * PI;
		switch (ax) {
		case X:
			r00 = 1.0f;
			r01 = 0;
			r02 = 0;
			r10 = 0;
			r11 = cos(a);
			r12 = (-1.0)*sin(a);
			r20 = 0;
			r21 = sin(a);
			r22 = cos(a);
			break;
		case Y:
			r00 = cos(a);
			r01 = 0;
			r02 = sin(a);
			r10 = 0;
			r11 = 1.0f;
			r12 = 0;
			r20 = (-1.0)*sin(a);
			r21 = 0;
			r22 = cos(a);
			break;
		case Z:
			r00 = cos(a);
			r01 = (-1.0)*sin(a);
			r02 = 0;
			r10 = sin(a);
			r11 = cos(a);
			r12 = 0;
			r20 = 0;
			r21 = 0;
			r22 = 1.0f;
			break;
		}
	}

	Vector operator*(Vector v) {
		return Vector(r11*v.x + r01*v.y + r02*v.z, r10*v.x + r11*v.y + r12*v.z, r20*v.x + r21*v.y + r22*v.z);
	}
};

class Intersectable {
protected:
	Material* material;

public:
	Intersectable(Material* m) : material(m) {

	}

	virtual Hit intersect(const Ray& ray) = 0;
};

class Ellipszoid : public Intersectable {
	Vector center;
	Vector nCenter;
	Vector speed;
	float xDist;
	float yDist;
	float zDist;
	RotationMatrix rY;
	RotationMatrix rZ;

	RotationMatrix nRz;
	RotationMatrix nRy;

public:
	Ellipszoid(Vector c, float x, float y, float z, Material* m, Vector s) : nCenter(c), speed(s), xDist(x), yDist(y), zDist(z), Intersectable(m), rZ(RotationMatrix(-45, Z)), rY(RotationMatrix(-45, Y)), nRz(RotationMatrix(45, Z)), nRy(RotationMatrix(45, Y)) {
		center = rZ*c;
		center = rY*center;
	}

	Hit intersect(const Ray& ray) {
		Vector p = ray.p;
		Vector dir = ray.dir;

		Vector op = ray.p;
		Vector oDir = ray.dir;

		p = rZ*p;
		dir = rZ*dir;
		p = rY*p;
		dir = rY*dir;
		
		Vector transformedSpeed = rZ*speed;
		transformedSpeed = rY*transformedSpeed;

		Hit hit;

		float a_x = (pow((dir.x + transformedSpeed.x), 2) / pow(2*xDist, 2));
		float a_y = (pow((dir.y + transformedSpeed.y), 2) / pow(2*yDist, 2));
		float a_z = (pow((dir.z + transformedSpeed.z), 2) / pow(2*zDist, 2));
		float a = a_x + a_y + a_z;
		float b_x = (2 * p.x*transformedSpeed.x + 2 * pow(transformedSpeed.x, 2)*ray.prevTime + 2 * dir.x*p.x + 2 * dir.x*transformedSpeed.x*ray.prevTime - 2 * center.x*transformedSpeed.x - 2 * center.x*dir.x) / pow(2*xDist, 2);
		float b_y = (2 * p.y*transformedSpeed.y + 2 * pow(transformedSpeed.y, 2)*ray.prevTime + 2 * dir.y*p.y + 2 * dir.y*transformedSpeed.y*ray.prevTime - 2 * center.y*transformedSpeed.y - 2 * center.y*dir.y) / pow(2*yDist, 2);
		float b_z = (2 * p.z*transformedSpeed.z + 2 * pow(transformedSpeed.z, 2)*ray.prevTime + 2 * dir.z*p.z + 2 * dir.z*transformedSpeed.z*ray.prevTime - 2 * center.z*transformedSpeed.z - 2 * center.z*dir.z) / pow(2*zDist, 2);
		float b = b_x + b_y + b_z;
		float c_x = (pow(center.x, 2) + pow(p.x, 2) + pow(transformedSpeed.x*ray.prevTime, 2) + 2 * p.x*transformedSpeed.x*ray.prevTime - 2 * center.x*p.x - 2 * center.x*transformedSpeed.x*ray.prevTime) / pow(2*xDist, 2);
		float c_y = (pow(center.y, 2) + pow(p.y, 2) + pow(transformedSpeed.y*ray.prevTime, 2) + 2 * p.y*transformedSpeed.y*ray.prevTime - 2 * center.y*p.y - 2 * center.y*transformedSpeed.y*ray.prevTime) / pow(2*yDist, 2);
		float c_z = (pow(center.z, 2) + pow(p.z, 2) + pow(transformedSpeed.z*ray.prevTime, 2) + 2 * p.z*transformedSpeed.z*ray.prevTime - 2 * center.z*p.z - 2 * center.z*transformedSpeed.z*ray.prevTime) / pow(2*zDist, 2);
		float c = c_x + c_y + c_z - 1;

		float D = pow(b, 2) - (4 * a*c);
		if (D < 0) {
			return hit;
		}
		float t1 = ((-1)*b + sqrt(D)) / (2 * a);

		hit.position = op + oDir*t1;

		Vector n =(nCenter - speed*(ray.prevTime) - speed*(t1)) - hit.position;
		n = rZ*n;
		n = rY*n;
		n = Vector(n.x / pow(xDist, 2), n.y / pow(yDist, 2), n.z / pow(zDist, 2));
		n = nRy*n;
		n = nRz*n;
		hit.t = t1;
		hit.material = material;
		hit.normal = n.normalize();

		return hit;
	}
};

class Paraboloid : public Intersectable {
	Vector center;

public:
	
	Paraboloid(Vector c, Material* m) : center(c), Intersectable(m) {

	}

	Hit intersect(const Ray& ray) {
		Vector p = ray.p;
		Vector dir = ray.dir;

		Hit hit;
		float a = pow(dir.x, 2) + pow(dir.z, 2);
		float b = 2 * (p.x*dir.x - center.x*dir.x + p.z*dir.z - center.z*dir.z) + dir.y;
		float c = pow(p.x - center.x, 2) + pow(p.z - center.z, 2) + p.y - center.y;
		float D = pow(b, 2) - (4 * a*c);
		if (D < 0) {
			return hit;
		}
		float t1 = ((-1)*b - sqrt(D)) / (2 * a);
		
		hit.t = t1;
		hit.material = material;
		hit.position = p + dir*t1;
		Vector n(2 * (hit.position.x - center.x), 1.0f, 2 * (hit.position.z - center.z));
		hit.normal = n.normalize();
		return hit;
	}
};

class Wall : public Intersectable {
	Vector p1;
	Vector p2;
	Vector p3;
	Vector p4;
	Vector n;

public:

	Wall(Vector c1, Vector c2, Vector c3, Vector c4, Material* m) : p1(c1), p2(c2), p3(c3), p4(c4), Intersectable(m) {
		n = ((c2 - c1) % (c3 - c1)).normalize();
	}

	Hit intersect(const Ray& ray) {
		Vector p = ray.p;
		Vector dir = ray.dir;
		Hit hit;
		float t1 = ((p1 - p)*n) / (dir*n);
		if (t1 <= 0) {
			return hit;
		}
		hit.t = t1;
		hit.material = material;
		hit.position = p + dir*t1;
		hit.normal = n;
		return hit;
	}
};

class Camera {
	Vector eye;
	Vector lookat;
	Vector up;
	Vector right;
	int XM;
	int YM;
public:
	Camera() : eye(5.0f, 5.0f, -5.0f), lookat(5.0f, 5.0f, 0.0f), up(0.0f, 5.0f, 0.0f), right(5.0f, 0.0f, 0.0f), XM(screenWidth), YM(screenHeight) {

	}

	Ray getRay(int x, int y, float ellapsed) {
		Ray ray;
		Vector p = lookat + right * (((2.0 * x) / XM) - 1) + up * (((2.0 * y) / YM) - 1);
		Vector dir = p - eye;
		ray.p = p;
		ray.dir = dir.normalize();
		ray.eye = eye;
		ray.prevTime = ellapsed;
		return ray;
	}
};

class Light {
	Vector p;
	Vector speed;
	Color Lout;

public:
	Light() {

	}

	Light(Vector position, Vector s) : Lout(Color(15.0f, 15.0f, 15.0f)), p(position), speed(s) {

	}

	Vector getLightDir(Vector v, float prevT) {
		return (getPosition(v, prevT) - v).normalize();
	}

	Color getInRad(Vector v, float prevT) {
		return Lout*(1 / pow((getPosition(v, prevT) - v).Length(), 2));
	}

	float getDist(Vector v) {
		return (p - v).Length();
	}

	Vector getPosition(Vector in, float prevT) {
		Vector tempP;
		Vector pp;
		if (prevT <= 0) {
			pp = p - speed*prevT;
		}
		else {
			pp = p;
		}

		float a = pow(speed.x, 2) + pow(speed.y, 2) + pow(speed.z, 2) - 1;
		float b = 2 * pp.x*speed.x - 2 * speed.x*in.x + 2 * pp.y*speed.y - 2 * speed.y*in.y + 2 * pp.z*speed.z - 2 * speed.z*in.z;
		float c = pow(pp.x - in.x, 2) + pow(pp.y - in.y, 2) + pow(pp.z - in.z, 2);
		float D = pow(b, 2) - 4 * a*c;
		float x = ((-1)*b - sqrt(D)) / (2 * a);

		tempP = pp + speed*x;

		return tempP;
	}
};

int maxDepth = 7;

class Scene {
	Intersectable* objects[9];
	int addedObj;
	Material* materials[7];
	Light lights;
	Camera camera;
public:
	Scene() : addedObj(0) {
		
	}

	void add(Intersectable* obj) {
		objects[addedObj] = obj;
		addedObj++;
	}

	void build() {
		Vector ellipseSpeed(-1.0f, 1.0f, 0.25f);
		ellipseSpeed = ellipseSpeed.normalize();
		ellipseSpeed = ellipseSpeed / 2;

		Color glassF0((pow(1.5 - 1, 2) / pow(1.5 + 1, 2)), (pow(1.5 - 1, 2) / pow(1.5 + 1, 2)), (pow(1.5 - 1, 2) / pow(1.5 + 1, 2)));
		Material* ellipseMat = new SmoothMaterial(1.5, glassF0, true, true);
		
		materials[0] = ellipseMat;
		Intersectable* e = new Ellipszoid(Vector(8.5f, 1.5f, 1.5f), 1.0f, 0.25f, 0.5f, ellipseMat, ellipseSpeed);

		Color goldF0((pow(0.17 - 1, 2) + 3.1) / (pow(0.17 + 1, 2) + 3.1), (pow(0.35 - 1, 2) + 2.7) / (pow(0.35 + 1, 2) + 2.7), (pow(1.5 - 1, 2) + 1.9) / (pow(1.5 + 1, 2) + 1.9));
		Material* paraboloidMat = new SmoothMaterial(1, goldF0, true, false);
		materials[1] = paraboloidMat;
		Intersectable* paraboloid = new Paraboloid(Vector(5.0f, 35.0f, 10.0f), paraboloidMat);

		Material* wall1Mat = new RoughMaterial(Color(0.627f, 0.0f, 0.627f), Color(1.0f, 1.0f, 1.0f), 35.0f);
		materials[2] = wall1Mat;
		Intersectable* w1 = new Wall(Vector(0, 0, 0), Vector(0, 10, 0), Vector(0, 0, 10), Vector(0, 10, 10), wall1Mat);
		Material* wall2Mat = new RoughMaterial(Color(0.0f, 0.627f, 0.0f), Color(1.0f, 1.0f, 1.0f), 35.0f);
		materials[3] = wall2Mat;
		Intersectable* w2 = new Wall(Vector(0, 0, 0), Vector(0, 0, 10), Vector(10, 0, 0), Vector(10, 0, 10), wall2Mat);
		Material* wall3Mat = new RoughMaterial(Color(0.831f, 0.541f, 0.0f), Color(1.0f, 1.0f, 1.0f), 35.0f);
		materials[4] = wall3Mat;
		Intersectable* w3 = new Wall(Vector(10, 0, 10), Vector(10, 10, 10), Vector(10, 0, 0), Vector(10, 10, 0), wall3Mat);
		Material* wall4Mat = new RoughMaterial(Color(0.0f, 0.733f, 0.831f), Color(1.0f, 1.0f, 1.0f), 35.0f);
		materials[5] = wall4Mat;
		Intersectable* w4 = new Wall(Vector(0, 10, 10), Vector(0, 10, 0), Vector(10, 10, 10), Vector(10, 10, 0), wall4Mat);
		Material* wall5Mat = new RoughMaterial(Color(0.831f, 0.0f, 0.0f), Color(1.0f, 1.0f, 1.0f), 35.0f);
		materials[6] = wall5Mat;
		Intersectable* w5 = new Wall(Vector(10, 10, 0), Vector(0, 10, 0), Vector(10, 0, 0), Vector(0, 0, 0), wall5Mat);

		add(e);
		add(w1);
		add(w2);
		add(w3);
		add(w4);
		add(w5);
		add(paraboloid);

		Light l(Vector(8.5f, 7.5f, 2.5f), Vector(-0.25f, 0.0f, 0.0f));
		lights = l;
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (int objNum = 0; objNum < addedObj; ++objNum) {
			Hit hit = objects[objNum]->intersect(ray);
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t)) {
				bestHit = hit;
			}
		}
		return bestHit;
	}

	Color trace(Ray ray, int depth) {
		Vector nullpos = ray.p;
		Vector nulldir = ray.dir;
		if (depth == maxDepth) {
			return Color(0, 0, 0);
		}
		Hit hit = firstIntersect(ray);
		if (hit.t < 0) {
			return Color(0, 0, 0);
		}
		Color outRadiance(0, 0, 0);
		Ray shadowRay(hit.position + (hit.normal * 0.001f), lights.getPosition(hit.position, ray.prevTime), true, ray.prevTime + hit.t);
		Hit shadowHit = firstIntersect(shadowRay);
		Vector tD = hit.position - Vector(5.0f, 5.0f, 5.0f);
		if (shadowHit.t < 0 || (shadowHit.t >(hit.position - lights.getPosition(hit.position, ray.prevTime)).Length())) {
			outRadiance += hit.material->shade(ray.dir, hit.normal, lights.getLightDir(hit.position, ray.prevTime), lights.getInRad(hit.position, ray.prevTime), tD.Length());
		}
		outRadiance += Color(0.08f, 0.08f, 0.08f) * hit.material->getColor(tD.Length());
		Vector inDir = (ray.p - hit.position).normalize();
		Vector outDir = (hit.position - ray.p).normalize();
		if (hit.material->isReflective()) {
			Vector reflectionDir = hit.material->reflect(outDir, hit.normal);
			Ray reflectionRay(hit.position + (hit.normal * 0.001f), reflectionDir, false, ray.prevTime + hit.t);
			outRadiance += trace(reflectionRay, ++depth) * hit.material->fresnel(inDir, hit.normal);
		}
		if (hit.material->isRefractive()) {
			Vector refractionDir = hit.material->refract(ray.dir, hit.normal);
			Ray refractedRay(hit.position - (hit.normal*0.001f), refractionDir, false, ray.prevTime + hit.t);
			outRadiance += trace(refractedRay, +depth) * (Color(1, 1, 1) - hit.material->fresnel(inDir, hit.normal));
		}
		return outRadiance;
	}

	~Scene() {
		for (int i = 0; i < addedObj; ++i) {
			delete objects[i];
		}
		for (int j = 0; j < 7; ++j) {
			delete materials[j];
		}
	}
};

Scene scene;
Color traced[screenWidth*screenHeight];
Camera cam;

void onInitialization( ) { 
	glViewport(0, 0, screenWidth, screenHeight);

    for(int Y = 0; Y < screenHeight; Y++)
		for(int X = 0; X < screenWidth; X++)
			traced[Y*screenWidth + X] = Color(0, 0, 0);

	scene.build();
}

void onDisplay( ) {
    glClearColor(0.1f, 0.2f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glDrawPixels(screenWidth, screenHeight, GL_RGB, GL_FLOAT, traced);

    glutSwapBuffers();

}

void onKeyboard(unsigned char key, int x, int y) {
    if (key == 'd') glutPostRedisplay( );

	if (key == ' ') {
		float time = (-1.0)*(float)glutGet(GLUT_ELAPSED_TIME) / 1000 - 3;
		for (int y = 0; y < screenHeight; ++y) {
			for (int x = 0; x < screenWidth; ++x) {
				traced[y*screenWidth + x] = scene.trace(cam.getRay(x, y, time), 1);
			}
		}

		glutPostRedisplay();
	}

}

void onKeyboardUp(unsigned char key, int x, int y) {

}

void onMouse(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN)
		glutPostRedisplay( ); 						 
}

void onMouseMotion(int x, int y)
{

}

void onIdle( ) {
     long time = glutGet(GLUT_ELAPSED_TIME);

}

// ...Idaig modosithatod
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// A C++ program belepesi pontja, a main fuggvenyt mar nem szabad bantani
int main(int argc, char **argv) {
    glutInit(&argc, argv); 				// GLUT inicializalasa
    glutInitWindowSize(600, 600);			// Alkalmazas ablak kezdeti merete 600x600 pixel 
    glutInitWindowPosition(100, 100);			// Az elozo alkalmazas ablakhoz kepest hol tunik fel
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);	// 8 bites R,G,B,A + dupla buffer + melyseg buffer

    glutCreateWindow("Grafika hazi feladat");		// Alkalmazas ablak megszuletik es megjelenik a kepernyon

    glMatrixMode(GL_MODELVIEW);				// A MODELVIEW transzformaciot egysegmatrixra inicializaljuk
    glLoadIdentity();
    glMatrixMode(GL_PROJECTION);			// A PROJECTION transzformaciot egysegmatrixra inicializaljuk
    glLoadIdentity();

    onInitialization();					// Az altalad irt inicializalast lefuttatjuk

    glutDisplayFunc(onDisplay);				// Esemenykezelok regisztralasa
    glutMouseFunc(onMouse); 
    glutIdleFunc(onIdle);
    glutKeyboardFunc(onKeyboard);
    glutKeyboardUpFunc(onKeyboardUp);
    glutMotionFunc(onMouseMotion);

    glutMainLoop();					// Esemenykezelo hurok
    
    return 0;
}

