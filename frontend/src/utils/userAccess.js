export function isAdminUser(user) {
  return user?.is_superuser === true
}

export function isUserRoleResolved(user) {
  return Boolean(user?.is_demo) || typeof user?.is_superuser === 'boolean'
}

export function getDefaultRouteForUser(user) {
  return isAdminUser(user) ? '/admin/dashboard' : '/app/coach'
}
