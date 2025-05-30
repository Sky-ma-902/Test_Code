#include "Player.h"
#include <QGraphicsScene>
#include <QKeyEvent>
#include "Bullet.h"
#include "Enemy.h"
#include <QAudioOutput>

Player::Player(QGraphicsItem *parent): QGraphicsPixmapItem(parent){
    // set bullet sound
    bulletsound = new QMediaPlayer();
    audioOutput = new QAudioOutput();
    bulletsound->setAudioOutput(audioOutput);
    audioOutput->setVolume(1.0);
    bulletsound->setSource(QUrl("qrc:/sounds/bullet.wav"));

    // set graphic
    setPixmap(QPixmap(":/images/player.png"));
}

void Player::keyPressEvent(QKeyEvent *event){
    // move the player left and right
    if (event->key() == Qt::Key_Left){
        if (pos().x() > 0)
            setPos(x()-10,y());
    }
    else if (event->key() == Qt::Key_Right){
        if (pos().x() + 100 < 800)
            setPos(x()+10,y());
    }
    // shoot with the spacebar
    else if (event->key() == Qt::Key_Space){
        // create a bullet
        Bullet * bullet = new Bullet();
        bullet->setPos(x()+45,y());
        scene()->addItem(bullet);

        // play bulletsound
        if (bulletsound->playbackState() == QMediaPlayer::PlayingState){
            bulletsound->setPosition(0);
        }
        else if (bulletsound->playbackState() == QMediaPlayer::StoppedState){
            bulletsound->play();
        }

    }
}

void Player::spawn(){
    // create an enemy
    Enemy * enemy = new Enemy();
    scene()->addItem(enemy);
}
