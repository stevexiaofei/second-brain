// Requires Playwright plus an installed Chromium-family browser. Does not install dependencies.
// Start site: npm run docs:dev -- --host 127.0.0.1 --port 5187 --strictPort
// Optional env: PLAYWRIGHT_MODULE_PATH, IO_DEMO_BASE_URL, IO_DEMO_BROWSER_PATH, IO_DEMO_SCREENSHOT_DIR.
import assert from 'node:assert/strict'
import { mkdir } from 'node:fs/promises'
import { createRequire } from 'node:module'
import path from 'node:path'
import { getIoScenario, IO_DEMOS, STATE_FIELDS } from '../docs/.vitepress/components/diskIoReadWriteModel.mjs'

const require = createRequire(import.meta.url)
const { chromium } = require(process.env.PLAYWRIGHT_MODULE_PATH || 'playwright')
const baseUrl = process.env.IO_DEMO_BASE_URL || 'http://127.0.0.1:5187'
const output = path.resolve(process.env.IO_DEMO_SCREENSHOT_DIR || '_tmp_screenshot')
await mkdir(output, { recursive: true })
const browser = await chromium.launch({ headless: true, ...(process.env.IO_DEMO_BROWSER_PATH ? { executablePath: process.env.IO_DEMO_BROWSER_PATH } : {}) })
const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } })
// The site has no configured favicon. Suppress only the browser's incidental default request.
await context.route('**/favicon.ico', route => route.fulfill({ status: 204, body: '' }))
const page = await context.newPage()
const errors = []
const consoleErrors = []
const failedResources = []
page.on('pageerror', error => errors.push(error.message))
page.on('response', response => { if (response.status() >= 400) failedResources.push(`${response.status()} ${response.url()}`) })
page.on('console', message => { if (message.type() === 'error') consoleErrors.push({ text: message.text(), location: message.location() }) })
let frameCount = 0
const root = () => page.locator('.io-demo').first()
const next = () => root().getByRole('button', { name: '下一步', exact: true })
const previous = () => root().getByRole('button', { name: '上一步', exact: true })

async function chooseMode(mode) {
  await root().locator('.mode-select select').selectOption(mode)
  await page.waitForFunction(mode => document.querySelector('.io-demo')?.dataset.ioMode === mode, mode)
}
async function assertFrame(scenario, index) {
  const expected = scenario.steps[index]
  const title = `${index + 1}. ${expected.title}`
  await page.waitForFunction(title => document.querySelector('.io-demo .stage-announcement h4')?.textContent === title, title)
  assert.equal(await root().locator('.api-code code').textContent(), expected.code)
  const rows = await root().locator('.state-table tbody tr').evaluateAll(rows => rows.map(row => row.lastElementChild.textContent))
  assert.deepEqual(rows, STATE_FIELDS.map(([id]) => expected.state[id]))
  frameCount++
}
async function jumpLast() {
  const select = root().locator('.step-jump select')
  const last = await select.locator('option').last().getAttribute('value')
  await select.selectOption(last)
}
async function screenshot(name) {
  // VitePress has a sticky nav. Hide it only in capture, not in interaction or layout assertions.
  await root().screenshot({ path: path.join(output, name), animations: 'disabled', style: '.VPNav, .VPLocalNav { visibility: hidden !important; }' })
}
async function assertNoPageOverflow(width) {
  const dimensions = await page.evaluate(() => ({ width: document.documentElement.clientWidth, content: document.documentElement.scrollWidth }))
  assert.equal(dimensions.width, width)
  assert.ok(dimensions.content <= width + 1, `Whole page overflow: ${JSON.stringify(dimensions)}`)
}

try {
  await page.goto(`${baseUrl}/knowledge/engineering/profiling/08-disk-io-benchmark.html`, { waitUntil: 'domcontentloaded', timeout: 120000 })
  await root().waitFor({ timeout: 120000 })
  assert.equal(await page.locator('.io-demo').count(), 5)
  for (const [mode, demo] of Object.entries(IO_DEMOS)) {
    await chooseMode(mode)
    for (const variant of demo.variants) {
      await root().getByRole('button', { name: variant.label, exact: true }).click()
      const scenario = getIoScenario(mode, variant.id)
      await assertFrame(scenario, 0)
      assert.equal(await previous().isDisabled(), true)
      for (let i = 1; i < scenario.steps.length; i++) {
        await next().click()
        await assertFrame(scenario, i)
      }
      assert.equal(await next().isDisabled(), true)
      await previous().click()
      await assertFrame(scenario, scenario.steps.length - 2)
      await root().getByRole('button', { name: '重置', exact: true }).click()
      await assertFrame(scenario, 0)
    }
  }
  for (const backend of ['local', 'network']) {
    for (const api of ['readv', 'preadv']) {
      for (const variant of ['hit', 'miss', 'readahead']) {
        await chooseMode('read')
        await root().getByRole('button', { name: IO_DEMOS.read.variants.find(v => v.id === variant).label, exact: true }).click()
        if (variant !== 'readahead') await root().locator('.read-options select').nth(0).selectOption(api)
        await root().locator('.read-options select').nth(1).selectOption(backend)
        const scenario = getIoScenario('read', variant, { api, backend })
        await jumpLast()
        await assertFrame(scenario, scenario.steps.length - 1)
      }
    }
  }
  await chooseMode('open')
  const role = root().getByRole('button', { name: /inode \/ 文件布局/ })
  await role.click()
  assert.match(await root().locator('.role-explanation').textContent(), /身份、权限、大小/)
  await role.click()
  assert.equal(await root().locator('.role-explanation').count(), 0)
  // Keyboard activation of a real button.
  await next().focus()
  await page.keyboard.press('Enter')
  await assertFrame(getIoScenario('open'), 1)

  await root().getByRole('button', { name: '重置', exact: true }).click()
  await root().locator('.controls select').selectOption('1600')
  await root().getByRole('button', { name: '播放', exact: true }).click()
  await page.waitForFunction(() => document.querySelector('.io-demo .stage-announcement h4')?.textContent.startsWith('2.'), { timeout: 10000 })
  await root().getByRole('button', { name: '暂停', exact: true }).click()
  const pausedTitle = await root().locator('.stage-announcement h4').textContent()
  await page.waitForTimeout(1800) // Intentional: verify that a paused timer no longer advances.
  assert.equal(await root().locator('.stage-announcement h4').textContent(), pausedTitle)
  await root().getByRole('button', { name: '播放', exact: true }).click()
  await root().getByRole('button', { name: 'dup 共享打开对象', exact: true }).click()
  await assertFrame(getIoScenario('open', 'dup'), 0)
  assert.equal(await root().getByRole('button', { name: '播放', exact: true }).count(), 1)
  await jumpLast()
  await root().getByRole('button', { name: '重播', exact: true }).click()
  await assertFrame(getIoScenario('open', 'dup'), 0)
  await root().getByRole('button', { name: '暂停', exact: true }).click()

  await chooseMode('write')
  await root().locator('.step-jump select').selectOption('3') // first write returned, dirty page visible
  await screenshot('disk-io-write-desktop.png')
  await chooseMode('read')
  await root().getByRole('button', { name: 'page cache 未命中', exact: true }).click()
  await root().locator('.read-options select').nth(1).selectOption('network')
  await root().locator('.step-jump select').selectOption('6')
  await screenshot('disk-io-network-read-desktop.png')
  await page.setViewportSize({ width: 390, height: 844 })
  await assertNoPageOverflow(390)
  await screenshot('disk-io-read-mobile-light.png')
  await page.setViewportSize({ width: 1440, height: 1000 })
  await page.locator('.VPSwitchAppearance:visible').first().click()
  await page.waitForFunction(() => document.documentElement.classList.contains('dark'))
  await page.setViewportSize({ width: 390, height: 844 })
  await screenshot('disk-io-read-mobile-dark.png')
  await assertNoPageOverflow(390)
  await page.setViewportSize({ width: 320, height: 700 })
  for (const mode of Object.keys(IO_DEMOS)) { await chooseMode(mode); await jumpLast(); await assertNoPageOverflow(320) }
  await page.emulateMedia({ reducedMotion: 'reduce' })
  await chooseMode('read')
  await next().click()
  assert.equal(await root().locator('.packet').first().evaluate(el => getComputedStyle(el).display), 'none')
  await page.emulateMedia({ reducedMotion: 'no-preference' })
  await page.setViewportSize({ width: 1440, height: 1000 })
  await page.goto(`${baseUrl}/knowledge/distributed-systems/distributed-storage/inode-file-metadata.html`, { waitUntil: 'domcontentloaded' })
  await root().waitFor()
  assert.equal(await page.locator('.io-demo').count(), 1)
  await root().getByRole('button', { name: 'dup 共享打开对象', exact: true }).click()
  await jumpLast()
  await assertFrame(getIoScenario('open', 'dup'), 6)
  assert.deepEqual(errors, [], 'Uncaught browser errors')
  assert.deepEqual(consoleErrors, [], 'Browser console errors')
  assert.deepEqual(failedResources, [], 'Application resource HTTP errors')
  console.log(`PASS: ${frameCount} browser frame checks; all branches, backwards/reset/jump, playback/pause/replay, keyboard, 320/390px layouts, dark mode, reduced motion, inode-page reuse; zero browser errors`)
  console.log(`Screenshots: ${output}`)
} finally {
  await browser.close()
}
