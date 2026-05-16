<?php
$config['imap_host'] = 'ssl://mailserver:993';
$config['imap_conn_options'] = array(
    'ssl' => array(
        'verify_peer' => true,
        'verify_peer_name' => true,
        'allow_self_signed' => false,
        'cafile' => '/var/roundcube/config/ca.crt',
    ),
);
$config['smtp_host'] = 'tls://mailserver:587';
$config['smtp_user'] = '%u';
$config['smtp_pass'] = '%p';
$config['smtp_conn_options'] = array(
    'ssl' => array(
        'verify_peer' => true,
        'verify_peer_name' => true,
        'allow_self_signed' => false,
        'cafile' => '/var/roundcube/config/ca.crt',
    ),
);
