from xml.dom.minidom import Document

from level.Level import Level


def create_camera(doc):
    camera = doc.createElement('Camera')
    camera.setAttribute('x', str(0))
    camera.setAttribute('y', str(1))
    camera.setAttribute('minWidth', str(20))
    camera.setAttribute('maxWidth', str(30))
    return camera


def create_birds(doc, level: Level, red_birds = False):
    birds = doc.createElement('Birds')
    if red_birds:
        for i in range(3):
            xml_bird = doc.createElement('Bird')
            xml_bird.setAttribute('type', 'BirdRed')
            birds.appendChild(xml_bird)
    else:
        for level_bird in level.birds:
            xml_bird = doc.createElement('Bird')
            xml_bird.setAttribute('type', str(level_bird.type))
            birds.appendChild(xml_bird)
    return birds


def create_slingshot(doc):
    slingshot = doc.createElement('Slingshot')
    slingshot.setAttribute('x', str(-8))
    slingshot.setAttribute('y', str(-2.5))
    return slingshot


def create_basis_level_node(level: Level = None, red_birds = False):
    doc = Document()
    doc.encoding = 'utf-8'
    node = doc.createElement('Level')
    node.setAttribute('width', str(2))
    node.appendChild(create_camera(doc))
    node.appendChild(create_birds(doc, level, red_birds))
    node.appendChild(create_slingshot(doc))

    return doc, node
